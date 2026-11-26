/*
 * record.h -- fights written down for the Leek Wars fight viewer.
 *
 * A recorded fight keeps its action log on, and empties it into a text after each
 * step, so that the log never reaches its capacity. Once the fight is over, the text is
 * the "data" object of a fight report, in the format of the Java generator
 * (Actions.toJSON): version, leeks, map, actions, dead and ops.
 */
#ifndef LEEKRL_RECORD_H
#define LEEKRL_RECORD_H

#include <stddef.h>

struct LwState;

typedef struct {
    char *data;
    size_t length, capacity;
    int failed;                 /* out of memory */
} leekrl_text;

typedef struct {
    leekrl_text map, leeks, actions;
    int n_leeks;                /* entities already written into leeks */
    unsigned char listed[64];   /* which fids are, one bit each */
    int n_actions;
    int overflow;               /* the action log was full: actions are missing */
    leekrl_text report;         /* the last fight, once over */
} leekrl_record;

/* After the fight has started. Writes the map and the first entities. */
void leekrl_record_start(leekrl_record *record, struct LwState *state);

/* Moves what the action log holds into the record, and empties the log. New entities
 * (summons) are written as they appear. */
void leekrl_record_drain(leekrl_record *record, struct LwState *state);

/* After the fight has finished: writes the report. 0 if the record is incomplete. */
int leekrl_record_finish(leekrl_record *record, struct LwState *state);

void leekrl_record_free(leekrl_record *record);

#endif
