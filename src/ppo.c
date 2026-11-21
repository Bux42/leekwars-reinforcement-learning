#include "leekrl/ppo.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct lr_ppo {
    size_t obs, hidden, actions, count;
    /* hidden weights, hidden biases, policy weights, policy biases,
       value weights, value bias */
    size_t hb, pw, pb, vw, vb;
    double *w;
};

typedef struct {
    size_t action;
    double logp, value, reward, next_value, advantage, target;
    int terminated, truncated;
} lr_step;

struct lr_trajectory {
    size_t capacity, obs, actions, size;
    double *observations;
    unsigned char *masks;
    lr_step *steps;
    int ready;
};

static int mul_size(size_t a, size_t b, size_t *out) {
    if (b && a > (size_t)-1 / b) return 0;
    *out = a * b;
    return 1;
}

static int add_size(size_t a, size_t b, size_t *out) {
    if (a > (size_t)-1 - b) return 0;
    *out = a + b;
    return 1;
}

static void *array_calloc(size_t count, size_t size) {
    size_t bytes;
    if (!mul_size(count, size, &bytes) || !bytes) return NULL;
    return calloc(count, size);
}

void lr_rng_seed(lr_rng *rng, uint64_t seed) {
    if (rng) rng->state = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
}

static uint64_t random_u64(lr_rng *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * UINT64_C(2685821657736338717);
}

static double uniform(lr_rng *rng) {
    return (double)(random_u64(rng) >> 11) * (1.0 / 9007199254740992.0);
}

lr_status lr_ppo_create(lr_ppo **out, size_t obs, size_t hidden,
                        size_t actions, lr_rng *rng) {
    lr_ppo *m;
    size_t hw, ph, i, count;
    double scale;
    if (!out) return LR_INVALID;
    *out = NULL;
    if (!obs || !hidden || !actions || !rng || !rng->state ||
        !mul_size(obs, hidden, &hw) || !mul_size(hidden, actions, &ph))
        return LR_INVALID;
    m = (lr_ppo *)calloc(1, sizeof(*m));
    if (!m) return LR_NOMEM;
    m->obs = obs; m->hidden = hidden; m->actions = actions;
    m->hb = hw;
    if (!add_size(m->hb, hidden, &m->pw) ||
        !add_size(m->pw, ph, &m->pb) ||
        !add_size(m->pb, actions, &m->vw) ||
        !add_size(m->vw, hidden, &m->vb) ||
        !add_size(m->vb, 1, &count) || count > (size_t)-1 / sizeof(double)) {
        free(m);
        return LR_INVALID;
    }
    m->count = count;
    m->w = (double *)array_calloc(count, sizeof(double));
    if (!m->w) { free(m); return LR_NOMEM; }
    scale = sqrt(2.0 / ((double)obs + (double)hidden));
    for (i = 0; i < hw; ++i) m->w[i] = (2.0 * uniform(rng) - 1.0) * scale;
    scale = sqrt(2.0 / ((double)hidden + (double)actions));
    for (i = m->pw; i < m->pb; ++i) m->w[i] = (2.0 * uniform(rng) - 1.0) * scale;
    scale = sqrt(2.0 / ((double)hidden + 1.0));
    for (i = m->vw; i < m->vb; ++i) m->w[i] = (2.0 * uniform(rng) - 1.0) * scale;
    *out = m;
    return LR_OK;
}

void lr_ppo_destroy(lr_ppo *m) {
    if (m) { free(m->w); free(m); }
}

size_t lr_ppo_observation_size(const lr_ppo *m) { return m ? m->obs : 0; }
size_t lr_ppo_action_count(const lr_ppo *m) { return m ? m->actions : 0; }

static lr_status value_forward(const lr_ppo *m, const double *obs,
                               double *hidden, double *value) {
    size_t i, j;
    double v;
    if (!m || !obs || !value) return LR_INVALID;
    for (j = 0; j < m->obs; ++j) if (!isfinite(obs[j])) return LR_INVALID;
    v = m->w[m->vb];
    for (i = 0; i < m->hidden; ++i) {
        double x = m->w[m->hb + i], h;
        for (j = 0; j < m->obs; ++j) x += m->w[i * m->obs + j] * obs[j];
        if (!isfinite(x)) return LR_NUMERIC;
        h = tanh(x);
        if (hidden) hidden[i] = h;
        v += m->w[m->vw + i] * h;
    }
    if (!isfinite(v)) return LR_NUMERIC;
    *value = v;
    return LR_OK;
}

lr_status lr_ppo_value(const lr_ppo *m, const double *obs, double *value) {
    return value_forward(m, obs, NULL, value);
}

/* logp is kept separately from probabilities: tiny probabilities can underflow. */
static lr_status forward(const lr_ppo *m, const double *obs,
                         const unsigned char *mask, double *h, double *p,
                         double *logp, double *value, double *entropy) {
    size_t i, j;
    double max_logit = -INFINITY, sum = 0.0, v, ent = 0.0, logsum;
    int any = 0;
    lr_status status;
    if (!m || !obs || !mask) return LR_INVALID;
    for (i = 0; i < m->actions; ++i) if (mask[i]) any = 1;
    if (!any) return LR_INVALID;
    status = value_forward(m, obs, h, &v);
    if (status != LR_OK) return status;
    for (i = 0; i < m->actions; ++i) {
        double x = m->w[m->pb + i];
        if (!mask[i]) { logp[i] = -INFINITY; continue; }
        for (j = 0; j < m->hidden; ++j) x += m->w[m->pw + i * m->hidden + j] * h[j];
        if (!isfinite(x)) return LR_NUMERIC;
        logp[i] = x;
        if (x > max_logit) max_logit = x;
    }
    for (i = 0; i < m->actions; ++i) {
        p[i] = mask[i] ? exp(logp[i] - max_logit) : 0.0;
        sum += p[i];
    }
    logsum = log(sum);
    for (i = 0; i < m->actions; ++i) {
        if (mask[i]) {
            logp[i] = (logp[i] - max_logit) - logsum;
            if (!isfinite(logp[i])) return LR_NUMERIC;
            if (p[i]) ent -= (p[i] / sum) * logp[i];
            p[i] /= sum;
        }
    }
    if (!isfinite(ent)) return LR_NUMERIC;
    *value = v; *entropy = ent;
    return LR_OK;
}

static lr_status eval_work(const lr_ppo *m, const double *obs,
                           const unsigned char *mask, double **work,
                           double *v, double *ent) {
    size_t n, two;
    lr_status status;
    if (!m || !obs || !mask || !mul_size(m->actions, 2, &two) ||
        !add_size(m->hidden, two, &n)) return LR_INVALID;
    *work = (double *)array_calloc(n, sizeof(double));
    if (!*work) return LR_NOMEM;
    status = forward(m, obs, mask, *work, *work + m->hidden,
                     *work + m->hidden + m->actions, v, ent);
    if (status != LR_OK) { free(*work); *work = NULL; }
    return status;
}

lr_status lr_ppo_evaluate(const lr_ppo *m, const double *obs,
                          const unsigned char *mask, size_t action,
                          double *probs, double *log_probability,
                          double *value, double *entropy) {
    double *work, v, ent;
    lr_status status;
    if (!m || !mask || action >= m->actions || !mask[action]) return LR_INVALID;
    status = eval_work(m, obs, mask, &work, &v, &ent);
    if (status != LR_OK) return status;
    if (probs) memcpy(probs, work + m->hidden, m->actions * sizeof(double));
    if (log_probability) *log_probability = work[m->hidden + m->actions + action];
    if (value) *value = v;
    if (entropy) *entropy = ent;
    free(work);
    return LR_OK;
}

lr_status lr_ppo_sample(const lr_ppo *m, const double *obs,
                        const unsigned char *mask, lr_rng *rng, size_t *action,
                        double *log_probability, double *value) {
    double *work, v, ent, r, cumulative = 0.0;
    size_t i, selected = 0;
    lr_status status;
    if (!rng || !rng->state || !action) return LR_INVALID;
    status = eval_work(m, obs, mask, &work, &v, &ent);
    if (status != LR_OK) return status;
    r = uniform(rng);
    /* Last nonzero-probability action is a rounding-safe fallback. */
    for (i = 0; i < m->actions; ++i) if (mask[i]) {
        if (work[m->hidden + i] > 0.0) selected = i;
        cumulative += work[m->hidden + i];
        if (r < cumulative) break;
    }
    *action = selected;
    if (log_probability) *log_probability = work[m->hidden + m->actions + selected];
    if (value) *value = v;
    free(work);
    return LR_OK;
}

lr_status lr_trajectory_create(lr_trajectory **out, size_t capacity,
                               size_t obs, size_t actions) {
    lr_trajectory *t;
    size_t no, na;
    if (!out) return LR_INVALID;
    *out = NULL;
    if (!capacity || !obs || !actions || !mul_size(capacity, obs, &no) ||
        !mul_size(capacity, actions, &na) ||
        no > (size_t)-1 / sizeof(double) ||
        capacity > (size_t)-1 / sizeof(lr_step)) return LR_INVALID;
    t = (lr_trajectory *)calloc(1, sizeof(*t));
    if (!t) return LR_NOMEM;
    t->observations = (double *)array_calloc(no, sizeof(double));
    t->masks = (unsigned char *)array_calloc(na, sizeof(unsigned char));
    t->steps = (lr_step *)array_calloc(capacity, sizeof(lr_step));
    if (!t->observations || !t->masks || !t->steps) {
        lr_trajectory_destroy(t); return LR_NOMEM;
    }
    t->capacity = capacity; t->obs = obs; t->actions = actions;
    *out = t;
    return LR_OK;
}

void lr_trajectory_destroy(lr_trajectory *t) {
    if (t) { free(t->observations); free(t->masks); free(t->steps); free(t); }
}
void lr_trajectory_clear(lr_trajectory *t) {
    if (t) { t->size = 0; t->ready = 0; }
}
size_t lr_trajectory_size(const lr_trajectory *t) { return t ? t->size : 0; }
size_t lr_trajectory_capacity(const lr_trajectory *t) { return t ? t->capacity : 0; }
size_t lr_trajectory_observation_size(const lr_trajectory *t) { return t ? t->obs : 0; }
size_t lr_trajectory_action_count(const lr_trajectory *t) { return t ? t->actions : 0; }

lr_status lr_trajectory_push(lr_trajectory *t, const double *obs,
                             const unsigned char *mask, size_t action,
                             double logp, double value, double reward,
                             double next_value, int terminated, int truncated) {
    size_t i;
    lr_step *s;
    if (!t || !obs || !mask || t->size == t->capacity ||
        action >= t->actions || !mask[action] ||
        !isfinite(logp) || logp > 0.0 || !isfinite(value) ||
        !isfinite(reward) || !isfinite(next_value) ||
        (terminated != 0 && terminated != 1) ||
        (truncated != 0 && truncated != 1))
        return LR_INVALID;
    for (i = 0; i < t->obs; ++i) if (!isfinite(obs[i])) return LR_INVALID;
    memcpy(t->observations + t->size * t->obs, obs, t->obs * sizeof(double));
    memcpy(t->masks + t->size * t->actions, mask, t->actions);
    s = t->steps + t->size++;
    s->action = action; s->logp = logp; s->value = value;
    s->reward = reward; s->next_value = next_value;
    s->terminated = terminated; s->truncated = truncated;
    t->ready = 0;
    return LR_OK;
}

lr_status lr_trajectory_gae(lr_trajectory *t, double gamma, double lambda) {
    size_t i;
    double carry = 0.0, *values;
    if (!t || !t->size || !isfinite(gamma) || !isfinite(lambda) ||
        gamma < 0.0 || gamma > 1.0 || lambda < 0.0 || lambda > 1.0)
        return LR_INVALID;
    values = (double *)array_calloc(t->size, sizeof(double));
    if (!values) return LR_NOMEM;
    for (i = t->size; i-- > 0;) {
        const lr_step *s = &t->steps[i];
        double delta = s->reward + (s->terminated ? 0.0 : gamma * s->next_value) - s->value;
        carry = delta + ((s->terminated || s->truncated) ? 0.0 : gamma * lambda * carry);
        if (!isfinite(carry) || !isfinite(carry + s->value)) {
            free(values); return LR_NUMERIC;
        }
        values[i] = carry;
    }
    for (i = 0; i < t->size; ++i) {
        t->steps[i].advantage = values[i];
        t->steps[i].target = values[i] + t->steps[i].value;
    }
    free(values); t->ready = 1;
    return LR_OK;
}

lr_status lr_trajectory_target(const lr_trajectory *t, size_t index,
                               double *advantage, double *target) {
    if (!t || !t->ready || index >= t->size || (!advantage && !target))
        return LR_INVALID;
    if (advantage) *advantage = t->steps[index].advantage;
    if (target) *target = t->steps[index].target;
    return LR_OK;
}

lr_status lr_ppo_update(lr_ppo *m, const lr_trajectory *t,
                        const lr_ppo_options *o) {
    size_t n, two, epoch, k, i, j;
    double *g, *work, *h, *p, *logp, *dh;
    lr_status status = LR_OK;
    if (!m || !t || !o || !t->ready || !t->size ||
        m->obs != t->obs || m->actions != t->actions ||
        !isfinite(o->learning_rate) || o->learning_rate <= 0.0 ||
        !isfinite(o->clip_ratio) || o->clip_ratio < 0.0 || o->clip_ratio >= 1.0 ||
        !isfinite(o->value_coefficient) || o->value_coefficient < 0.0 ||
        !isfinite(o->entropy_coefficient) || o->entropy_coefficient < 0.0 || !o->epochs ||
        !mul_size(m->actions, 2, &two) || !mul_size(m->hidden, 2, &n) ||
        !add_size(n, two, &n)) return LR_INVALID;
    g = (double *)array_calloc(m->count, sizeof(double));
    work = (double *)array_calloc(n, sizeof(double));
    if (!g || !work) { free(g); free(work); return LR_NOMEM; }
    h = work; p = h + m->hidden; logp = p + m->actions; dh = logp + m->actions;
    for (epoch = 0; epoch < o->epochs; ++epoch) {
        memset(g, 0, m->count * sizeof(double));
        for (k = 0; k < t->size; ++k) {
            const lr_step *s = &t->steps[k];
            const double *obs = t->observations + k * t->obs;
            const unsigned char *mask = t->masks + k * t->actions;
            double v, ent, factor = 0.0, vd, logratio;
            status = forward(m, obs, mask, h, p, logp, &v, &ent);
            if (status != LR_OK) goto done;
            logratio = logp[s->action] - s->logp;
            if (!isfinite(logratio)) { status = LR_NUMERIC; goto done; }
            if ((s->advantage > 0.0 && logratio <= log1p(o->clip_ratio)) ||
                (s->advantage < 0.0 && logratio >= log1p(-o->clip_ratio))) {
                factor = -s->advantage * exp(logratio);
                if (!isfinite(factor)) { status = LR_NUMERIC; goto done; }
            }
            vd = o->value_coefficient * (v - s->target);
            memset(dh, 0, m->hidden * sizeof(double));
            for (i = 0; i < m->actions; ++i) if (mask[i]) {
                double d = factor * ((i == s->action ? 1.0 : 0.0) - p[i]);
                if (p[i] > 0.0) d += o->entropy_coefficient * p[i] * (logp[i] + ent);
                g[m->pb + i] += d;
                for (j = 0; j < m->hidden; ++j) {
                    g[m->pw + i * m->hidden + j] += d * h[j];
                    dh[j] += d * m->w[m->pw + i * m->hidden + j];
                }
            }
            g[m->vb] += vd;
            for (j = 0; j < m->hidden; ++j) {
                g[m->vw + j] += vd * h[j];
                dh[j] = (dh[j] + vd * m->w[m->vw + j]) * (1.0 - h[j] * h[j]);
                g[m->hb + j] += dh[j];
                for (i = 0; i < m->obs; ++i)
                    g[j * m->obs + i] += dh[j] * obs[i];
            }
        }
        for (i = 0; i < m->count; ++i) {
            g[i] = g[i] / (double)t->size * o->learning_rate;
            if (!isfinite(g[i]) || !isfinite(m->w[i] - g[i])) {
                status = LR_NUMERIC; goto done;
            }
        }
        for (i = 0; i < m->count; ++i) m->w[i] -= g[i];
    }
done:
    free(g); free(work);
    return status;
}
