#include "leekrl/rules.h"

const int leekrl_solo_excluded_items[] = {
    408, /* weapon odachi: Nasu Samurai */
    409, /* weapon excalibur: Fennel King */
    410, /* weapon scythe: Evil Pumpkin */
    411, /* chip shuriken */
    412, /* chip kemuridama */
    413, /* chip fire_ball */
    414, /* chip trebuchet */
    415, /* chip awakening */
    416, /* chip thunder */
    417, /* chip kill */
    418, /* chip apocalypse */
    419, /* chip divine_protection */
    425, /* chip exasperation */
    448, /* chip grail_protection */
    84   /* chip resurrection: postponed, see specs/postponed/resurrection.md */
};

const size_t leekrl_solo_excluded_count =
    sizeof(leekrl_solo_excluded_items) / sizeof(leekrl_solo_excluded_items[0]);

int leekrl_solo_item_excluded(int item_id)
{
    size_t i;
    for (i = 0; i < leekrl_solo_excluded_count; ++i)
        if (leekrl_solo_excluded_items[i] == item_id) return 1;
    return 0;
}
