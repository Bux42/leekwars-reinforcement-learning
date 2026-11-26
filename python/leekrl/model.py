"""Policy and value network.

Most inputs and nearly all actions are zeros. The map and the legal actions are therefore
lists of indices.

The map has a fixed shape: the cells outside it only depend on where the acting entity
stands. The network gets that position, and only the cells that hold something.
"""

import torch
from torch import nn

from .vec import action_table

MAP_CELLS = 2381
MAP_RADIUS = 34
CATEGORIES = 8
FREE = 1
HELD = CATEGORIES - FREE - 1        # categories of the cells that hold something
PLACES = 35 ** 2                    # where the map can be

# Entity blocks in the floats: see specs/Observation.md.
ENTITY_START, ENTITY_SIZE, ENTITIES = 3, 116, 18
PRESENT, DX, DY = 0, 18, 19


def relative_cells():
    """dx and dy of each cell of the map observation."""
    rows = [(x, y) for y in range(-MAP_RADIUS, MAP_RADIUS + 1)
            for x in range(abs(y) - MAP_RADIUS, MAP_RADIUS - abs(y) + 1)]
    return torch.tensor(rows, dtype=torch.float32).unbind(1)


def map_index(dx, dy):
    """Position of relative cells in the map, for integer tensors."""
    radius = MAP_RADIUS
    above = (dy + radius) ** 2
    below = MAP_CELLS - (radius - dy + 1) ** 2
    return torch.where(dy <= 0, above, below) + dx + radius - dy.abs()


def gather(values, offsets, counts, picked, sizes):
    """The lists of some rows, from the lists of all rows, one after the other.

    offsets and counts tell where the list of each row is. sizes is counts[picked], on the
    CPU. Returns the lists, and for each of their values the row it comes from.
    """
    total = int(sizes.sum())
    chosen = counts[picked]
    rows = torch.repeat_interleave(torch.arange(len(picked), device=values.device), chosen,
                                   output_size=total)
    starts = chosen.cumsum(0) - chosen
    where = torch.arange(total, device=values.device) + (offsets[picked] - starts)[rows]
    return values[where], rows


class Legal:
    """The legal actions of a batch, as flat lists of (row, action) pairs, by row."""

    def __init__(self, actions, sizes, rows=None):
        """sizes are the numbers of legal actions, on the CPU."""
        device = actions.device
        self.batch = len(sizes)
        self.actions = actions.long()
        self.sizes = sizes.long()
        self.starts = self.sizes.cumsum(0) - self.sizes
        self.offsets = self.starts.to(device, non_blocking=True)
        if rows is None:
            rows = torch.repeat_interleave(torch.arange(self.batch, device=device),
                                           self.sizes.to(device, non_blocking=True),
                                           output_size=len(actions))
        self.rows = rows


class Inputs:
    """What the network receives for a batch of decisions."""

    def __init__(self, floats, place, held, held_sizes, legal, held_rows=None, sight=None,
                 sight_sizes=None):
        """held holds a code per cell that is not free, held_sizes their numbers, on the CPU.
        sight, if given, the cells from which the enemy leek is seen, and their numbers."""
        self.floats = floats
        self.place = place.long()
        self.held = held.long()
        self.held_sizes = held_sizes.long()
        self.held_offsets = (self.held_sizes.cumsum(0) - self.held_sizes).to(
            floats.device, non_blocking=True)
        self.held_rows = held_rows
        self.legal = legal
        self.sight = None if sight is None else sight.long()
        if sight is not None:
            sight_sizes = sight_sizes.long()
            self.sight_offsets = (sight_sizes.cumsum(0) - sight_sizes).to(floats.device,
                                                                          non_blocking=True)

    def cells(self):
        """Category of each cell. Those outside the map are said to be free."""
        device, batch = self.floats.device, len(self.floats)
        rows = self.held_rows
        if rows is None:
            rows = torch.repeat_interleave(torch.arange(batch, device=device),
                                           self.held_sizes.to(device),
                                           output_size=len(self.held))
        cells = torch.full((batch, MAP_CELLS), FREE, device=device)
        cells[rows, self.held // HELD] = self.held % HELD + FREE + 1
        return cells


class Distribution:
    """Probabilities over the legal actions of each row, from their flat logits."""

    def __init__(self, logits, legal):
        self.legal = legal
        rows, batch = legal.rows, legal.batch
        largest = torch.full((batch,), -torch.inf, dtype=logits.dtype, device=logits.device)
        largest = largest.scatter_reduce(0, rows, logits.detach(), "amax")
        shifted = logits - largest[rows]
        total = logits.new_zeros(batch).index_add(0, rows, shifted.exp())
        self.log_probs = shifted - total.log()[rows]
        self.probs = self.log_probs.exp()

    def entropy(self):
        zeros = self.log_probs.new_zeros(self.legal.batch)
        return -zeros.index_add(0, self.legal.rows, self.probs * self.log_probs)

    def sample(self, generator=None):
        """Positions in the flat lists, one per row, on the CPU."""
        # There: short lists, and floating-point cumsum is not deterministic on GPU.
        legal = self.legal
        cumulated = self.probs.detach().cpu().double().cumsum(0)
        ends = cumulated[legal.starts + legal.sizes - 1]
        starts = torch.cat([ends.new_zeros(1), ends[:-1]])
        uniform = torch.rand(legal.batch, dtype=torch.float64, generator=generator)
        position = torch.searchsorted(cumulated, starts + uniform * (ends - starts),
                                      right=True)
        # Rounding must never leave the row.
        return torch.minimum(torch.maximum(position, legal.starts),
                             legal.starts + legal.sizes - 1)

    def best(self):
        """Positions of the likeliest actions, the first one on ties, on the CPU."""
        legal = self.legal
        largest = torch.full((legal.batch,), -torch.inf, dtype=self.log_probs.dtype,
                             device=self.log_probs.device)
        largest = largest.scatter_reduce(0, legal.rows, self.log_probs, "amax")
        candidates = (self.log_probs == largest[legal.rows]).nonzero(as_tuple=True)[0]
        rows = legal.rows[candidates]
        first = torch.ones_like(rows, dtype=torch.bool)
        first[1:] = rows[1:] != rows[:-1]
        return candidates[first].cpu()


class FlatHead(nn.Module):
    """One independent output per action."""

    def __init__(self, hidden, actions):
        super().__init__()
        self.logits = nn.Linear(hidden, actions)
        # Nearly uniform at first.
        nn.init.normal_(self.logits.weight, std=0.001)
        nn.init.zeros_(self.logits.bias)

    def forward(self, hidden, inputs):
        # Every output is computed: gathering the weights of the legal ones only is no
        # faster, and takes far more memory.
        legal = inputs.legal
        return self.logits(hidden)[legal.rows, legal.actions]


class PairHead(nn.Module):
    """Scores an action from what it uses (item, move, end of turn) and its target cell.

    What is learned about a cell (its category, its place, the entity on it) serves every
    item, and what is learned about an item serves every cell.
    """

    def __init__(self, hidden, size=32):
        super().__init__()
        table = torch.from_numpy(action_table()).long()
        kind, item, dx, dy = table.unbind(1)
        weapons = int(item[kind == 2].max()) + 1
        # 0 ends the turn, 1 moves, then weapons, then chips.
        tool = torch.where(kind == 2, 2 + item, torch.where(kind == 3, 2 + weapons + item,
                                                            kind))
        self.register_buffer("tool", tool, persistent=False)
        self.register_buffer("cell", map_index(dx, dy), persistent=False)
        x, y = relative_cells()
        x, y = x / 10, y / 10
        places = torch.stack([x, y, x * x, y * y, x * y, x.abs() + y.abs()], 1)
        self.register_buffer("places", places, persistent=False)
        tools = int(tool.max()) + 1
        self.category = nn.Embedding(CATEGORIES, size)
        self.place = nn.Embedding(MAP_CELLS, size)
        self.geometry = nn.Linear(places.shape[1], size, bias=False)
        self.entity = nn.Linear(ENTITY_SIZE, size)
        self.fixed = nn.Embedding(tools, size)      # what the tool looks for, always
        self.adapted = nn.Embedding(tools, size)    # weighted by the situation
        self.weights = nn.Linear(hidden, size)
        self.shared = nn.Linear(hidden, size)       # what the situation looks for
        self.preference = nn.Linear(hidden, tools)  # of the tool, whatever the cell
        for module in (self.category, self.fixed, self.adapted):
            nn.init.normal_(module.weight, std=0.1)
        nn.init.normal_(self.place.weight, std=0.01)
        for module in (self.weights, self.shared, self.preference):
            nn.init.orthogonal_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.entity.weight, gain=0.1)
        nn.init.zeros_(self.entity.bias)
        nn.init.normal_(self.geometry.weight, std=0.1)

    def forward(self, hidden, inputs):
        legal = inputs.legal
        rows, tool, cell = legal.rows, self.tool[legal.actions], self.cell[legal.actions]
        batch = hidden.shape[0]
        blocks = inputs.floats[:, ENTITY_START:ENTITY_START + ENTITIES * ENTITY_SIZE]
        blocks = blocks.reshape(batch, ENTITIES, ENTITY_SIZE)
        present = blocks[:, :, PRESENT] > 0
        entities = self.entity(blocks)
        entities = torch.cat([entities.new_zeros(batch, 1, entities.shape[2]), entities], 1)
        where = map_index((blocks[:, :, DX] * MAP_RADIUS).round().long(),
                          (blocks[:, :, DY] * MAP_RADIUS).round().long())
        holder = torch.zeros(batch, MAP_CELLS, dtype=torch.long, device=hidden.device)
        row, slot = present.nonzero(as_tuple=True)
        holder[row, where[row, slot]] = slot + 1
        target = (self.category(inputs.cells()[rows, cell])
                  + self.place(cell) + self.geometry(self.places)[cell]
                  + entities[rows, holder[rows, cell]])
        wanted = (self.fixed(tool) + self.adapted(tool) * self.weights(hidden)[rows]
                  + self.shared(hidden)[rows])
        return (wanted * target).sum(1) + self.preference(hidden)[rows, tool]


class Policy(nn.Module):
    def __init__(self, actions, floats, branch=128, hidden=256, mask_input=True,
                 head="flat", layers=2, sight=False):
        super().__init__()
        self.settings = {"actions": actions, "floats": floats, "branch": branch,
                         "hidden": hidden, "mask_input": mask_input, "head": head,
                         "layers": layers, "sight": sight}
        # A vector per (cell, category) for obstacles and entities, and one per position
        # on the map: together they say as much as the whole map.
        self.map = nn.EmbeddingBag(MAP_CELLS * HELD, branch, mode="sum")
        self.position = nn.Embedding(PLACES, branch)
        # A vector per cell from which the enemy leek can be seen.
        self.sight = nn.EmbeddingBag(MAP_CELLS, branch, mode="sum") if sight else None
        self.floats = nn.Linear(floats, branch)
        # A vector per action, summed over the legal ones.
        self.mask = nn.EmbeddingBag(actions, branch, mode="sum") if mask_input else None
        width = branch * (3 if mask_input else 2)
        trunk = [nn.LayerNorm(width), nn.ReLU()]
        for i in range(layers):
            trunk += [nn.Linear(hidden if i else width, hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*trunk)
        if head == "flat":
            self.policy = FlatHead(hidden, actions)
        elif head == "pair":
            self.policy = PairHead(hidden)
        else:
            raise ValueError(f"unknown head {head!r}")
        self.value = nn.Linear(hidden, 1)
        nn.init.normal_(self.map.weight, std=0.1)
        nn.init.normal_(self.position.weight, std=0.1)
        if sight:
            nn.init.normal_(self.sight.weight, std=0.1)
        if mask_input:
            nn.init.normal_(self.mask.weight, std=0.06)
        nn.init.orthogonal_(self.value.weight, gain=1)
        nn.init.zeros_(self.value.bias)

    def _trunk(self, inputs):
        where = self.map(inputs.held, inputs.held_offsets) + self.position(inputs.place)
        if self.sight is not None:
            if inputs.sight is None:
                raise ValueError("this network needs the sight lists")
            where = where + self.sight(inputs.sight, inputs.sight_offsets)
        parts = [where, self.floats(inputs.floats)]
        if self.mask is not None:
            parts.append(self.mask(inputs.legal.actions, inputs.legal.offsets))
        return self.trunk(torch.cat(parts, 1))

    def forward(self, inputs):
        """Returns distribution and value."""
        hidden = self._trunk(inputs)
        logits = self.policy(hidden, inputs)
        return Distribution(logits, inputs.legal), self.value(hidden).squeeze(1)
