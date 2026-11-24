#include "leekrl/threads.h"

#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef SRWLOCK lock_t;
typedef CONDITION_VARIABLE condition_t;
typedef HANDLE thread_t;
#define lock_init(l) InitializeSRWLock(l)
#define lock_take(l) AcquireSRWLockExclusive(l)
#define lock_release(l) ReleaseSRWLockExclusive(l)
#define lock_free(l) ((void)(l))
#define condition_init(c) InitializeConditionVariable(c)
#define condition_wait(c, l) SleepConditionVariableSRW((c), (l), INFINITE, 0)
#define condition_wake_all(c) WakeAllConditionVariable(c)
#define condition_free(c) ((void)(c))
#else
#include <pthread.h>
typedef pthread_mutex_t lock_t;
typedef pthread_cond_t condition_t;
typedef pthread_t thread_t;
#define lock_init(l) pthread_mutex_init((l), NULL)
#define lock_take(l) pthread_mutex_lock(l)
#define lock_release(l) pthread_mutex_unlock(l)
#define lock_free(l) pthread_mutex_destroy(l)
#define condition_init(c) pthread_cond_init((c), NULL)
#define condition_wait(c, l) pthread_cond_wait((c), (l))
#define condition_wake_all(c) pthread_cond_broadcast(c)
#define condition_free(c) pthread_cond_destroy(c)
#endif

struct leekrl_threads {
    lock_t lock;
    condition_t start;
    condition_t finished;
    thread_t *workers;
    int n_workers;
    int stopping;
    unsigned generation; /* incremented for each run */
    int busy;            /* workers still in the current run */
    leekrl_task task;
    void *context;
    int count;
    int next;            /* first index not taken yet */
    int chunk;
};

/* Take and run chunks until none is left. */
static void work(leekrl_threads *threads)
{
    for (;;) {
        int first, last;
        lock_take(&threads->lock);
        first = threads->next;
        last = first + threads->chunk;
        if (last > threads->count) last = threads->count;
        threads->next = last;
        lock_release(&threads->lock);
        if (first >= last) return;
        for (; first < last; ++first) threads->task(threads->context, first);
    }
}

#ifdef _WIN32
static DWORD WINAPI worker(LPVOID argument)
#else
static void *worker(void *argument)
#endif
{
    leekrl_threads *threads = argument;
    unsigned seen = 0;
    for (;;) {
        lock_take(&threads->lock);
        while (!threads->stopping && threads->generation == seen)
            condition_wait(&threads->start, &threads->lock);
        if (threads->stopping) {
            lock_release(&threads->lock);
            return 0;
        }
        seen = threads->generation;
        lock_release(&threads->lock);
        work(threads);
        lock_take(&threads->lock);
        if (--threads->busy == 0) condition_wake_all(&threads->finished);
        lock_release(&threads->lock);
    }
}

leekrl_threads *leekrl_threads_create(int count)
{
    leekrl_threads *threads;
    if (count < 1 || count > 1024) return NULL;
    threads = calloc(1, sizeof(*threads));
    if (!threads) return NULL;
    lock_init(&threads->lock);
    condition_init(&threads->start);
    condition_init(&threads->finished);
    threads->workers = calloc((size_t)count, sizeof(*threads->workers));
    if (!threads->workers) {
        leekrl_threads_destroy(threads);
        return NULL;
    }
    while (threads->n_workers < count - 1) {
#ifdef _WIN32
        HANDLE handle = CreateThread(NULL, 0, worker, threads, 0, NULL);
        if (!handle) {
            leekrl_threads_destroy(threads);
            return NULL;
        }
        threads->workers[threads->n_workers++] = handle;
#else
        if (pthread_create(&threads->workers[threads->n_workers], NULL, worker, threads)) {
            leekrl_threads_destroy(threads);
            return NULL;
        }
        ++threads->n_workers;
#endif
    }
    return threads;
}

void leekrl_threads_destroy(leekrl_threads *threads)
{
    int i;
    if (!threads) return;
    lock_take(&threads->lock);
    threads->stopping = 1;
    condition_wake_all(&threads->start);
    lock_release(&threads->lock);
    for (i = 0; i < threads->n_workers; ++i) {
#ifdef _WIN32
        WaitForSingleObject(threads->workers[i], INFINITE);
        CloseHandle(threads->workers[i]);
#else
        pthread_join(threads->workers[i], NULL);
#endif
    }
    condition_free(&threads->start);
    condition_free(&threads->finished);
    lock_free(&threads->lock);
    free(threads->workers);
    free(threads);
}

void leekrl_threads_run(leekrl_threads *threads, leekrl_task task, void *context, int count)
{
    if (count <= 0) return;
    lock_take(&threads->lock);
    threads->task = task;
    threads->context = context;
    threads->count = count;
    threads->next = 0;
    threads->chunk = count / ((threads->n_workers + 1) * 8);
    if (threads->chunk < 1) threads->chunk = 1;
    threads->busy = threads->n_workers;
    ++threads->generation;
    condition_wake_all(&threads->start);
    lock_release(&threads->lock);
    work(threads);
    lock_take(&threads->lock);
    while (threads->busy) condition_wait(&threads->finished, &threads->lock);
    lock_release(&threads->lock);
}
