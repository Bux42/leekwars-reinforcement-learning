#ifndef LEEKRL_PPO_H
#define LEEKRL_PPO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* All arrays have the dimensions supplied at creation. Nonzero mask entries
 * mean valid actions. Functions return a status, never abort. */
typedef enum {
    LR_OK = 0, LR_INVALID, LR_NOMEM, LR_NUMERIC
} lr_status;

typedef struct { uint64_t state; } lr_rng;
typedef struct lr_ppo lr_ppo;
typedef struct lr_trajectory lr_trajectory;

/* Zero is a valid seed. Use separate RNGs for independent streams. */
void lr_rng_seed(lr_rng *rng, uint64_t seed);

/* One tanh hidden layer shared by linear policy and value heads.
 * The RNG initializes weights and is advanced on success. */
lr_status lr_ppo_create(lr_ppo **out, size_t observation_size,
                        size_t hidden_size, size_t action_count, lr_rng *rng);
void lr_ppo_destroy(lr_ppo *model);
size_t lr_ppo_observation_size(const lr_ppo *model);
size_t lr_ppo_action_count(const lr_ppo *model);

/* Evaluate the value head without requiring any legal actions. */
lr_status lr_ppo_value(const lr_ppo *model, const double *observation,
                       double *value);

/* Evaluate a valid action or sample from the masked policy. All-invalid masks
 * and non-finite observations are rejected. probs may be NULL; otherwise it
 * receives action_count probabilities, with exactly zero for invalid actions.
 * All scalar outputs may be NULL except action in sample. */
lr_status lr_ppo_evaluate(const lr_ppo *model, const double *observation,
                          const unsigned char *mask, size_t action,
                          double *probs, double *log_probability,
                          double *value, double *entropy);
lr_status lr_ppo_sample(const lr_ppo *model, const double *observation,
                        const unsigned char *mask, lr_rng *rng, size_t *action,
                        double *log_probability, double *value);

/* Storage copies observations and masks. Every step supplies the value of its
 * actual successor observation; for true termination this value is ignored.
 * Either flag ends GAE propagation across an episode boundary; truncation
 * bootstraps unless termination is also set.
 * A rollout ending without either flag also bootstraps but cannot propagate
 * GAE beyond the stored rollout. Values and log probabilities must be those
 * measured by the behavior policy during collection. */
lr_status lr_trajectory_create(lr_trajectory **out, size_t capacity,
                               size_t observation_size, size_t action_count);
void lr_trajectory_destroy(lr_trajectory *trajectory);
void lr_trajectory_clear(lr_trajectory *trajectory);
size_t lr_trajectory_size(const lr_trajectory *trajectory);
size_t lr_trajectory_capacity(const lr_trajectory *trajectory);
size_t lr_trajectory_observation_size(const lr_trajectory *trajectory);
size_t lr_trajectory_action_count(const lr_trajectory *trajectory);
lr_status lr_trajectory_push(lr_trajectory *trajectory,
                             const double *observation, const unsigned char *mask,
                             size_t action, double log_probability, double value,
                             double reward, double next_value,
                             int terminated, int truncated);

/* Computes unnormalized advantages and value targets (advantage + old value).
 * gamma and lambda must be in [0,1]. Push/clear invalidate these results. */
lr_status lr_trajectory_gae(lr_trajectory *trajectory, double gamma, double lambda);
lr_status lr_trajectory_target(const lr_trajectory *trajectory, size_t index,
                               double *advantage, double *target);

typedef struct {
    double learning_rate;  /* positive */
    double clip_ratio;     /* [0,1) */
    double value_coefficient;   /* nonnegative */
    double entropy_coefficient; /* nonnegative */
    size_t epochs;         /* positive */
} lr_ppo_options;

/* Full-batch gradient descent on clipped policy loss, squared value loss
 * (coefficient * 0.5 * error^2), and negative masked entropy. GAE must have
 * been computed since the last push/clear. Each epoch is applied atomically;
 * an error in a later epoch does not roll back earlier epochs. */
lr_status lr_ppo_update(lr_ppo *model, const lr_trajectory *trajectory,
                        const lr_ppo_options *options);

#ifdef __cplusplus
}
#endif
#endif
