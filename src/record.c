#include "leekrl/record.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lw_actions.h"
#include "lw_cell.h"
#include "lw_chip.h"
#include "lw_constants.h"
#include "lw_entity.h"
#include "lw_map.h"
#include "lw_state.h"
#include "lw_weapon.h"

/* ---- Texts ------------------------------------------------------------- */

static void reserve(leekrl_text *text, size_t more)
{
    size_t capacity;
    char *data;
    if (text->failed || text->length + more + 1 <= text->capacity) return;
    capacity = text->capacity ? text->capacity : 4096;
    while (capacity < text->length + more + 1) capacity *= 2;
    data = realloc(text->data, capacity);
    if (!data) {
        text->failed = 1;
        return;
    }
    text->data = data;
    text->capacity = capacity;
}

static void add(leekrl_text *text, const char *format, ...)
{
    va_list args;
    int n;
    reserve(text, 64);
    if (text->failed) return;
    va_start(args, format);
    n = vsnprintf(text->data + text->length, text->capacity - text->length, format, args);
    va_end(args);
    if (n < 0) {
        text->failed = 1;
        return;
    }
    if ((size_t)n >= text->capacity - text->length) {
        reserve(text, (size_t)n);
        if (text->failed) return;
        va_start(args, format);
        vsnprintf(text->data + text->length, text->capacity - text->length, format, args);
        va_end(args);
    }
    text->length += (size_t)n;
}

static void add_text(leekrl_text *text, const leekrl_text *other)
{
    if (other->failed) text->failed = 1;
    if (!other->length) return;
    reserve(text, other->length);
    if (text->failed) return;
    memcpy(text->data + text->length, other->data, other->length);
    text->data[text->length += other->length] = 0;
}

static void add_string(leekrl_text *text, const char *s)
{
    add(text, "\"");
    for (; s && *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') add(text, "\\%c", c);
        else if (c < 0x20) add(text, "\\u%04x", c);
        else add(text, "%c", c);
    }
    add(text, "\"");
}

static void clear(leekrl_text *text)
{
    text->length = 0;
    text->failed = 0;
    if (text->data) text->data[0] = 0;
}

static void release(leekrl_text *text)
{
    free(text->data);
    memset(text, 0, sizeof(*text));
}

/* ---- Fight data ---------------------------------------------------------- */

/* Java Actions.addMap. Maps without an id have no custom obstacles. */
static void write_map(leekrl_text *text, LwMap *map)
{
    int i, first = 1;
    if (map->id) add(text, "{\"id\":%d,\"obstacles\":{", map->id);
    else add(text, "{\"obstacles\":{");
    for (i = 0; i < map->nb_cells; ++i) {
        const LwCell *cell = &map->cells[i];
        if (cell->walkable || lw_cell_get_obstacle_size(cell) <= 0) continue;
        add(text, "%s\"%d\":%d", first ? "" : ",", cell->id,
            map->id ? lw_cell_get_obstacle(cell) : lw_cell_get_obstacle_size(cell));
        first = 0;
    }
    /* The height is the width, as in Java. */
    add(text, "},\"type\":%d,\"width\":%d,\"height\":%d}", map->type, map->width, map->width);
}

/* Java Actions.addEntity. */
static void write_entity(leekrl_text *text, LwEntity *e, int critical)
{
    const LwCell *cell = lw_entity_get_cell(e);
    int i;
    if (text->length) add(text, ",");
    add(text, "{\"id\":%d,\"level\":%d,\"skin\":%d,", lw_entity_get_fid(e),
        lw_entity_get_level(e), lw_entity_get_skin(e));
    if (lw_entity_get_hat(e) > 0) add(text, "\"hat\":%d,", lw_entity_get_hat(e));
    else add(text, "\"hat\":null,");
    add(text, "\"metal\":%s,\"face\":%d,\"life\":%d,", lw_entity_get_metal(e) ? "true" : "false",
        lw_entity_get_face(e), lw_entity_get_life(e));
    add(text, "\"strength\":%d,\"wisdom\":%d,\"agility\":%d,\"resistance\":%d,"
              "\"frequency\":%d,\"science\":%d,\"magic\":%d,",
        lw_entity_get_stat(e, LW_STAT_STRENGTH), lw_entity_get_stat(e, LW_STAT_WISDOM),
        lw_entity_get_stat(e, LW_STAT_AGILITY), lw_entity_get_stat(e, LW_STAT_RESISTANCE),
        lw_entity_get_stat(e, LW_STAT_FREQUENCY), lw_entity_get_stat(e, LW_STAT_SCIENCE),
        lw_entity_get_stat(e, LW_STAT_MAGIC));
    add(text, "\"cores\":%d,\"ram\":%d,\"tp\":%d,\"mp\":%d,\"team\":%d,\"name\":",
        lw_entity_get_cores(e), lw_entity_get_ram(e), lw_entity_get_tp(e),
        lw_entity_get_mp(e), lw_entity_get_team(e) + 1);
    add_string(text, lw_entity_get_name(e));
    if (cell) add(text, ",\"cellPos\":%d", cell->id);
    else add(text, ",\"cellPos\":null");
    add(text, ",\"farmer\":%d,\"type\":%d,\"orientation\":%d,\"summon\":%s",
        lw_entity_get_farmer(e), lw_entity_get_type(e), lw_entity_get_orientation(e),
        lw_entity_is_summon(e) ? "true" : "false");
    if (lw_entity_is_summon(e)) {
        LwEntity *owner = lw_entity_get_summoner(e);
        add(text, ",\"owner\":%d,\"critical\":%s", owner ? lw_entity_get_fid(owner) : -1,
            critical ? "true" : "false");
    }
    if (lw_entity_get_type(e) == LW_ENTITY_TYPE_LEEK) {
        add(text, ",\"equipment\":{\"weapons\":[");
        for (i = 0; i < lw_entity_get_weapons_count(e); ++i)
            add(text, "%s%d", i ? "," : "", lw_weapon_get_id(lw_entity_get_weapon_at(e, i)));
        add(text, "],\"chips\":[");
        for (i = 0; i < lw_entity_get_chips_count(e); ++i)
            add(text, "%s%d", i ? "," : "", lw_chip_get_id(lw_entity_get_chip_at(e, i)));
        add(text, "],\"components\":[]}");
    }
    add(text, "}");
}

/* Whether an entity is written already, by fid. */
static int listed(leekrl_record *record, int fid, int set)
{
    static const int size = (int)(sizeof(record->listed) * 8);
    if (fid < 0 || fid >= size) return 1;
    if (set) record->listed[fid / 8] |= (unsigned char)(1 << (fid % 8));
    return (record->listed[fid / 8] >> (fid % 8)) & 1;
}

/* Whether a summon was a critical hit, from the actions of the log. */
static int critical_summon(const LwActionStream *stream, int fid)
{
    int i;
    for (i = 0; i < stream->n_entries; ++i) {
        const LwActionLog *entry = &stream->entries[i];
        if (entry->type == LW_ACTION_SUMMON && entry->n_args >= 4 && entry->v[1] == fid)
            return entry->v[3] == LW_ATTACK_USE_CRITICAL;
    }
    return 0;
}

static void write_new_entities(leekrl_record *record, LwState *state, int initial)
{
    const LwActionStream *stream = &state->actions.stream;
    int n = initial ? state->n_initial_order : state->n_entities, i;
    for (i = 0; i < n; ++i) {
        LwEntity *e = initial ? state->initial_order[i] : state->m_entities[i];
        int fid;
        if (!e) continue;
        fid = lw_entity_get_fid(e);
        if (listed(record, fid, 0)) continue;
        listed(record, fid, 1);
        write_entity(&record->leeks, e, critical_summon(stream, fid));
        ++record->n_leeks;
    }
}

/* ---- Public functions ----------------------------------------------------- */

void leekrl_record_start(leekrl_record *record, LwState *state)
{
    clear(&record->map);
    clear(&record->leeks);
    clear(&record->actions);
    memset(record->listed, 0, sizeof(record->listed));
    record->n_leeks = record->n_actions = record->overflow = 0;
    write_map(&record->map, state->map);
    write_new_entities(record, state, 1);
    leekrl_record_drain(record, state);
}

void leekrl_record_drain(leekrl_record *record, LwState *state)
{
    LwActionStream *stream = &state->actions.stream;
    leekrl_text *text = &record->actions;
    int i, j;
    if (stream->n_entries >= LW_LOG_MAX_ENTRIES || stream->n_extra >= LW_LOG_MAX_EXTRA)
        record->overflow = 1;
    write_new_entities(record, state, 0);
    for (i = 0; i < stream->n_entries; ++i) {
        const LwActionLog *entry = &stream->entries[i];
        add(text, "%s[%d", record->n_actions++ ? "," : "", entry->type);
        for (j = 0; j < entry->n_args; ++j) add(text, ",%d", entry->v[j]);
        if (entry->extra_offset >= 0) {
            add(text, ",[");
            for (j = 0; j < entry->extra_len; ++j)
                add(text, "%s%d", j ? "," : "", stream->extra[entry->extra_offset + j]);
            add(text, "]");
        }
        add(text, "]");
    }
    stream->n_entries = stream->n_extra = 0;
}

int leekrl_record_finish(leekrl_record *record, LwState *state)
{
    leekrl_text *report = &record->report;
    int i, first;
    leekrl_record_drain(record, state);
    clear(report);
    add(report, "{\"version\":1,\"leeks\":[");
    add_text(report, &record->leeks);
    add(report, "],\"map\":");
    add_text(report, &record->map);
    add(report, ",\"actions\":[");
    add_text(report, &record->actions);
    add(report, "],\"dead\":{");
    /* By id. Summons take the id of their summoner: only the others. */
    for (i = 0, first = 1; i < state->n_entities; ++i) {
        LwEntity *e = state->m_entities[i];
        if (lw_entity_is_summon(e)) continue;
        add(report, "%s\"%d\":%s", first ? "" : ",", lw_entity_get_id(e),
            lw_entity_is_dead(e) ? "true" : "false");
        first = 0;
    }
    add(report, "},\"ops\":{");
    for (i = 0; i < state->actions.n_operations; ++i)
        add(report, "%s\"%d\":%lld", i ? "," : "", state->actions.operations[i].entity_id,
            (long long)state->actions.operations[i].operations);
    add(report, "}}");
    return !report->failed && !record->overflow;
}

void leekrl_record_free(leekrl_record *record)
{
    release(&record->map);
    release(&record->leeks);
    release(&record->actions);
    release(&record->report);
}
