"""
scenarios.py — Pre-built world scenarios.

Each scenario has a name, description, and a factory function that
returns a fresh World ready to play.  Scenarios are intentionally
varied to stress-test different agent capabilities:

  key_door      — navigation + item use (find key → unlock door → reach goal)
  gem_collector — coverage + collection (visit all gems on a larger map)
  maze          — pure pathfinding (no items, just navigation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .world import Cell, Terrain, World


@dataclass
class Scenario:
    name: str
    description: str
    build: Callable[[], World]


# ── Scenario 1: Key & Door ────────────────────────────────────────────────────
#
#  9 × 7 grid.  Agent starts top-left.
#  Key is in the storage room.  Locked door separates corridor from exit.
#
#  # # # # # # # # #
#  # @ . . . . . . #
#  # . k . . . . . #
#  # . . . . . . . #
#  # . . . . . . . #
#  # . . . . D . G #
#  # # # # # # # # #


def _build_key_door() -> World:
    w = World(
        width=9,
        height=7,
        goal=(
            "Find the key (k), unlock the locked door (D), "
            "and step onto the goal tile (G)."
        ),
        max_steps=40,
    )

    # Outer walls
    for x in range(9):
        w.grid[0][x].terrain = Terrain.WALL
        w.grid[6][x].terrain = Terrain.WALL
    for y in range(7):
        w.grid[y][0].terrain = Terrain.WALL
        w.grid[y][8].terrain = Terrain.WALL

    # Key at (2, 2)
    w.grid[2][2].item = "key"
    w.grid[2][2].label = "Storage room"

    # Locked door at (5, 5), goal at (7, 5)
    w.grid[5][5].terrain = Terrain.DOOR_LOCKED
    w.grid[5][7].terrain = Terrain.GOAL
    w.grid[5][7].label = "Exit"

    # Labels
    w.grid[1][1].label = "Entrance"
    w.grid[3][4].label = "Main corridor"

    w.agent_pos = (1, 1)
    return w


# ── Scenario 2: Gem Collector ─────────────────────────────────────────────────
#
#  11 × 9 grid.  Three gems hidden in different rooms.
#  Agent must collect all gems.  Interior walls force detours.
#
#  # # # # # # # # # # #
#  # @ . . # . * . . . #
#  # . . . # . . . . . #
#  # . . . # . . . . . #
#  # . . . . . . . . . #
#  # . * . . . . # . . #
#  # . . . . . . # . * #
#  # . . . . . . # . . #
#  # # # # # # # # # # #


def _build_gem_collector() -> World:
    w = World(
        width=11,
        height=9,
        goal="Collect all 3 gems (* symbols) scattered around the map.",
        max_steps=60,
        total_gems=3,
    )

    # Outer walls
    for x in range(11):
        w.grid[0][x].terrain = Terrain.WALL
        w.grid[8][x].terrain = Terrain.WALL
    for y in range(9):
        w.grid[y][0].terrain = Terrain.WALL
        w.grid[y][10].terrain = Terrain.WALL

    # Vertical wall x=4, y=1..3  (blocks direct east path from start)
    for y in range(1, 4):
        w.grid[y][4].terrain = Terrain.WALL

    # Vertical wall x=7, y=5..7  (creates right-side pocket)
    for y in range(5, 8):
        w.grid[y][7].terrain = Terrain.WALL

    # Gems
    w.grid[1][6].item = "gem"   # gem 1: top-right area
    w.grid[5][2].item = "gem"   # gem 2: middle-left
    w.grid[6][9].item = "gem"   # gem 3: bottom-right pocket

    # Labels
    w.grid[1][1].label = "Starting room"
    w.grid[1][6].label = "East wing"
    w.grid[5][2].label = "West corridor"
    w.grid[6][9].label = "Back room"

    w.agent_pos = (1, 1)
    return w


# ── Scenario 3: Maze ──────────────────────────────────────────────────────────
#
#  String-defined maze.  No items — pure navigation.
#  Agent must find a path from @ to G.


def _build_maze() -> World:
    layout = [
        "# # # # # # # # # # # # #",
        "# @ . . . . . . . . . . #",
        "# . # # # . # # # # # . #",
        "# . # . . . . . # . . . #",
        "# . # . # # # . # . # # #",
        "# . . . # . . . . . # . #",
        "# # # # # . # # # # # . #",
        "# . . . . . # . . . . . #",
        "# . # # # . # . # # # . #",
        "# . . . . . . . . . . . G",
        "# # # # # # # # # # # # #",
    ]

    rows = [row.split() for row in layout]
    height = len(rows)
    width = len(rows[0])

    w = World(
        width=width,
        height=height,
        goal="Navigate the maze and reach the exit tile (G).",
        max_steps=120,
    )

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                w.grid[y][x].terrain = Terrain.WALL
            elif ch == "G":
                w.grid[y][x].terrain = Terrain.GOAL
                w.grid[y][x].label = "Exit"
            elif ch == "@":
                w.agent_pos = (x, y)

    return w


# ── Registry ──────────────────────────────────────────────────────────────────

SCENARIOS: list[Scenario] = [
    Scenario(
        name="key_door",
        description="Classic: find the key, unlock the door, reach the goal.",
        build=_build_key_door,
    ),
    Scenario(
        name="gem_collector",
        description="Collect all 3 gems scattered across a larger map.",
        build=_build_gem_collector,
    ),
    Scenario(
        name="maze",
        description="Navigate a maze to reach the exit — no items, pure pathfinding.",
        build=_build_maze,
    ),
]

SCENARIO_MAP = {s.name: s for s in SCENARIOS}
