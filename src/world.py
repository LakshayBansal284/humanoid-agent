"""
world.py — Grid world engine.

Defines the World state, Cell, Terrain types, and Item definitions.
The World is the single source of truth; the LLM agent never touches
it directly — it only reads Observations and calls Actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ── Terrain ───────────────────────────────────────────────────────────────────


class Terrain(Enum):
    FLOOR = "floor"
    WALL = "wall"
    DOOR_LOCKED = "door_locked"
    DOOR_OPEN = "door_open"
    GOAL = "goal"


TERRAIN_SYMBOLS: Dict[Terrain, str] = {
    Terrain.FLOOR: ".",
    Terrain.WALL: "#",
    Terrain.DOOR_LOCKED: "D",
    Terrain.DOOR_OPEN: "_",
    Terrain.GOAL: "G",
}

TERRAIN_NAMES: Dict[Terrain, str] = {
    Terrain.FLOOR: "floor",
    Terrain.WALL: "solid wall",
    Terrain.DOOR_LOCKED: "locked door",
    Terrain.DOOR_OPEN: "open doorway",
    Terrain.GOAL: "GOAL TILE — step here to win!",
}


# ── Items ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ItemDef:
    name: str
    symbol: str
    pickable: bool
    description: str


ITEM_DEFS: Dict[str, ItemDef] = {
    "key": ItemDef(
        "key", "k", pickable=True,
        description="A small iron key — use it on an adjacent locked door",
    ),
    "gem": ItemDef(
        "gem", "*", pickable=True,
        description="A sparkling gem — collect all gems to win",
    ),
    "torch": ItemDef(
        "torch", "t", pickable=True,
        description="A lit torch casting warm light",
    ),
    "crate": ItemDef(
        "crate", "C", pickable=False,
        description="A heavy wooden crate blocking the path",
    ),
    "note": ItemDef(
        "note", "n", pickable=True,
        description="A folded piece of paper with writing on it",
    ),
}


# ── World ─────────────────────────────────────────────────────────────────────


@dataclass
class Cell:
    terrain: Terrain = Terrain.FLOOR
    item: Optional[str] = None   # key into ITEM_DEFS
    label: Optional[str] = None  # human-readable place name


@dataclass
class World:
    """
    The entire game state lives here.  The agent harness reads
    this to build observations and writes it via action execution.
    """

    width: int
    height: int
    goal: str = ""

    # Mutable state
    grid: List[List[Cell]] = field(default_factory=list)
    agent_pos: Tuple[int, int] = (1, 1)
    inventory: List[str] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 50
    done: bool = False
    success: bool = False
    event_log: List[str] = field(default_factory=list)

    # Gem-collection bookkeeping
    total_gems: int = 0
    gems_collected: int = 0

    def __post_init__(self) -> None:
        if not self.grid:
            self.grid = [
                [Cell() for _ in range(self.width)]
                for _ in range(self.height)
            ]

    # ── Queries ───────────────────────────────────────────────────────────────

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def cell_at(self, x: int, y: int) -> Cell:
        return self.grid[y][x]

    def is_passable(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        t = self.cell_at(x, y).terrain
        return t not in (Terrain.WALL, Terrain.DOOR_LOCKED)

    def count_items_on_map(self, item_name: str) -> int:
        return sum(
            1 for row in self.grid for c in row if c.item == item_name
        )

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def log(self, msg: str) -> None:
        """Append to the event log (keep last 8 entries)."""
        self.event_log.append(msg)
        if len(self.event_log) > 8:
            self.event_log = self.event_log[-8:]

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Return an ASCII art map of the world."""
        rows: List[str] = []
        for y in range(self.height):
            chars: List[str] = []
            for x in range(self.width):
                if (x, y) == self.agent_pos:
                    chars.append("@")
                else:
                    c = self.cell_at(x, y)
                    if c.item:
                        idef = ITEM_DEFS.get(c.item)
                        chars.append(idef.symbol if idef else "?")
                    else:
                        chars.append(TERRAIN_SYMBOLS[c.terrain])
            rows.append(" ".join(chars))
        return "\n".join(rows)
