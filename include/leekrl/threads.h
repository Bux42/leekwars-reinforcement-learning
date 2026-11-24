#ifndef LEEKRL_THREADS_H
#define LEEKRL_THREADS_H

#ifdef __cplusplus
extern "C" {
#endif

/* A pool of threads running the same task on indices 0 to count - 1. */
typedef struct leekrl_threads leekrl_threads;
typedef void (*leekrl_task)(void *context, int index);

/* count includes the calling thread. NULL on failure. */
leekrl_threads *leekrl_threads_create(int count);
void leekrl_threads_destroy(leekrl_threads *threads);

/* Returns once every index is done. One call at a time. */
void leekrl_threads_run(leekrl_threads *threads, leekrl_task task, void *context, int count);

#ifdef __cplusplus
}
#endif
#endif
