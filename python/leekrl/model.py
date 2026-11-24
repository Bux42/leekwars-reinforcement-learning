"""Policy and value network.

Most inputs and nearly all actions are zeros. Maps and masks are therefore handled as
lists of indices, which is several times faster on CPU than dense arrays.

The map has a fixed shape: the cells outside it only depend on where the acting entity
stands. The network gets that position, and only the cells that hold something.
"""

import torch
from torch import nn

from .vec import action_table

MAP_CELLS = 2381
MAP_RADIUS = 34
REAL_CELLS = 613
CATEGORIES = 8
FREE = 1

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


class Legal:
    """The legal actions of a batch, as flat lists of (row, action) pairs, by row."""

    def __init__(self, mask):
        self.rows, self.actions = mask.nonzero(as_tuple=True)
        # Counting the pairs is much faster than summing the mask.
        self.counts = torch.bincount(self.rows, minlength=mask.shape[0])
        self.offsets = torch.cat([self.counts.new_zeros(1), self.counts.cumsum(0)[:-1]])
        self.batch = mask.shape[0]


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
        """Positions in the flat lists, one per row."""
        # On CPU: short lists, and floating-point cumsum is not deterministic on GPU.
        legal = self.legal
        offsets, counts = legal.offsets.cpu(), legal.counts.cpu()
        cumulated = self.probs.detach().cpu().double().cumsum(0)
        ends = cumulated[offsets + counts - 1]
        starts = torch.cat([ends.new_zeros(1), ends[:-1]])
        uniform = torch.rand(legal.batch, dtype=torch.float64, generator=generator)
        position = torch.searchsorted(cumulated, starts + uniform * (ends - starts),
                                      right=True)
        # Rounding must never leave the row.
        position = torch.minimum(torch.maximum(position, offsets), offsets + counts - 1)
        return position.to(self.probs.device)

    def best(self):
        """Positions of the likeliest actions, the first one on ties."""
        legal = self.legal
        largest = torch.full((legal.batch,), -torch.inf, dtype=self.log_probs.dtype,
                             device=self.log_probs.device)
        largest = largest.scatter_reduce(0, legal.rows, self.log_probs, "amax")
        candidates = (self.log_probs == largest[legal.rows]).nonzero(as_tuple=True)[0]
        rows = legal.rows[candidates]
        first = torch.ones_like(rows, dtype=torch.bool)
        first[1:] = rows[1:] != rows[:-1]
        return candidates[first]

    def find(self, actions):
        """Positions of given actions, one per row. They must be legal."""
        legal = self.legal
        keys = legal.rows * (1 << 32) + legal.actions
        wanted = torch.arange(legal.batch, device=actions.device) * (1 << 32) + actions
        position = torch.searchsorted(keys, wanted)
        if not torch.equal(keys[position.clamp(max=len(keys) - 1)], wanted):
            raise ValueError("illegal action")
        return position


class FlatHead(nn.Module):
    """One independent output per action."""

    def __init__(self, hidden, actions):
        super().__init__()
        self.logits = nn.Linear(hidden, actions)
        # Nearly uniform at first.
        nn.init.normal_(self.logits.weight, std=0.001)
        nn.init.zeros_(self.logits.bias)

    def forward(self, hidden, cells, floats, legal):
        # Every output is computed: gathering the weights of the legal ones only is no
        # faster, and takes far more memory.
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

    def forward(self, hidden, cells, floats, legal):
        rows, tool, cell = legal.rows, self.tool[legal.actions], self.cell[legal.actions]
        batch = hidden.shape[0]
        blocks = floats[:, ENTITY_START:ENTITY_START + ENTITIES * ENTITY_SIZE]
        blocks = blocks.reshape(batch, ENTITIES, ENTITY_SIZE)
        present = blocks[:, :, PRESENT] > 0
        entities = self.entity(blocks)
        entities = torch.cat([entities.new_zeros(batch, 1, entities.shape[2]), entities], 1)
        where = map_index((blocks[:, :, DX] * MAP_RADIUS).round().long(),
                          (blocks[:, :, DY] * MAP_RADIUS).round().long())
        holder = torch.zeros(batch, MAP_CELLS, dtype=torch.long, device=hidden.device)
        row, slot = present.nonzero(as_tuple=True)
        holder[row, where[row, slot]] = slot + 1
        target = (self.category(cells[rows, cell].long())
                  + self.place(cell) + self.geometry(self.places)[cell]
                  + entities[rows, holder[rows, cell]])
        wanted = (self.fixed(tool) + self.adapted(tool) * self.weights(hidden)[rows]
                  + self.shared(hidden)[rows])
        return (wanted * target).sum(1) + self.preference(hidden)[rows, tool]


class Policy(nn.Module):
    def __init__(self, actions, floats, branch=128, hidden=256, mask_input=True,
                 head="flat", layers=2):
        super().__init__()
        self.settings = {"actions": actions, "floats": floats, "branch": branch,
                         "hidden": hidden, "mask_input": mask_input, "head": head,
                         "layers": layers}
        # A vector per (cell, category) for obstacles and entities, and one per position
        # on the map: together they say as much as the whole map.
        self.map = nn.EmbeddingBag(MAP_CELLS * (CATEGORIES - FREE - 1), branch, mode="sum")
        self.position = nn.Embedding((MAP_RADIUS + 1) ** 2, branch)
        x, y = relative_cells()
        self.register_buffer("offsets", torch.stack([x, y], 1), persistent=False)
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
        if mask_input:
            nn.init.normal_(self.mask.weight, std=0.06)
        nn.init.orthogonal_(self.value.weight, gain=1)
        nn.init.zeros_(self.value.bias)

    def place(self, cells):
        """Index of the acting entity's position, from the center of the map seen by it."""
        # The map is symmetric: its center is the mean of its cells.
        center = ((cells != 0).float() @ self.offsets / REAL_CELLS).round().long()
        half = MAP_RADIUS // 2
        return (center[:, 1] + half) * (MAP_RADIUS + 1) + center[:, 0] + half

    def _trunk(self, cells, floats, legal):
        held = cells > FREE
        rows, columns = held.nonzero(as_tuple=True)
        indices = columns * (CATEGORIES - FREE - 1) + cells[rows, columns].long() - FREE - 1
        counts = torch.bincount(rows, minlength=cells.shape[0])
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        parts = [self.map(indices, offsets) + self.position(self.place(cells)),
                 self.floats(floats)]
        if self.mask is not None:
            parts.append(self.mask(legal.actions, legal.offsets))
        return self.trunk(torch.cat(parts, 1))

    def forward(self, cells, floats, mask):
        """cells: uint8 categories; floats; mask: bool. Returns distribution and value."""
        legal = Legal(mask)
        hidden = self._trunk(cells, floats, legal)
        logits = self.policy(hidden, cells, floats, legal)
        return Distribution(logits, legal), self.value(hidden).squeeze(1)
