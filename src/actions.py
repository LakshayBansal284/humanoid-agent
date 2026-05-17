"""
actions.py — Action space and execution engine.

Two things live here:
  1. ACTION_TOOLS  — the Claude tool-use schema given to the LLM
  2. execute()     — turns a (name, params) pair into a world mutation + message

The LLM never calls execute() directly; the harness (agent.py) does.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .world import ITEM_DEFS, TERRAIN_NAMES, Cell, Terrain, World

# ── Cardinal directions ───────────────────────────────────────────────────────

DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "north": (0, -1),
    "south": (0,  1),
    "east":  (1,  0),
    "west":  (-1, 0),
}

# ── Tool schemas (given verbatim to the Claude API) ───────────────────────────

ACTION_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "move",
        "description": (
            "Move the agent one step in the given direction. "
            "Fails if a wall, locked door, or impassable cell blocks the way."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["north", "south", "east", "west"],
                    "description": "Cardinal direction to move.",
                }
            },
            "required": ["direction"],
        },
    },
    {
        "name": "pick_up",
        "description": (
            "Pick up the item at the agent's current position and add it to inventory. "
            "Fails if there is no pickable item here."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "use_item",
        "description": (
            "Use an item from inventory on the current cell or an adjacent cell. "
            "Example: use 'key' on 'north' to unlock a door to the north."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": "Name of the item in inventory to use.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["north", "south", "east", "west", "here"],
                    "description": (
                        "Direction of the target cell. Use 'here' to act on the current cell."
                    ),
                },
            },
            "required": ["item", "direction"],
        },
    },
    {
        "name": "look",
        "description": (
            "Carefully examine the current cell and all four adjacent cells. "
            "Returns a detailed description including terrain, items, and passability."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "declare_done",
        "description": (
            "Signal that you believe the goal is complete. "
            "Use this for collection goals (e.g. all gems gathered). "
            "For navigation goals, simply step onto the GOAL tile instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why the goal is complete.",
                }
            },
            "required": ["reason"],
        },
    },
]


# ── Action execution ──────────────────────────────────────────────────────────


def execute(world: World, action_name: str, params: Dict[str, Any]) -> str:
    """
    Apply action to world; return a plain-English result string.
    The world is mutated in-place.
    """
    if action_name == "move":
        return _move(world, params["direction"])
    if action_name == "pick_up":
        return _pick_up(world)
    if action_name == "use_item":
        return _use_item(world, params["item"], params.get("direction", "here"))
    if action_name == "look":
        return _look(world)
    if action_name == "declare_done":
        # Caller (agent.py) decides whether this counts as success
        world.done = True
        msg = f"You declare the task complete: {params.get('reason', '...')}"
        world.log(msg)
        return msg
    return f"Unknown action '{action_name}'."


# ── Individual action implementations ────────────────────────────────────────


def _move(world: World, direction: str) -> str:
    dx, dy = DIRECTIONS.get(direction, (0, 0))
    ax, ay = world.agent_pos
    nx, ny = ax + dx, ay + dy

    if not world.in_bounds(nx, ny):
        return f"You can't move {direction} — that's out of bounds."

    cell = world.cell_at(nx, ny)

    if cell.terrain == Terrain.WALL:
        return f"A solid wall blocks your path to the {direction}."
    if cell.terrain == Terrain.DOOR_LOCKED:
        return (
            f"A locked door blocks your path to the {direction}. "
            "You'll need a key to open it."
        )

    world.agent_pos = (nx, ny)
    world.step_count += 1

    # ── Goal tile ──
    if cell.terrain == Terrain.GOAL:
        world.done = True
        world.success = True
        msg = "🎉 You step onto the GOAL tile — task complete!"
        world.log(msg)
        return msg

    # ── Build result message ──
    parts = [f"You move {direction} to ({nx}, {ny})."]
    if cell.label:
        parts.append(f"You are now in: {cell.label}.")
    if cell.item:
        idef = ITEM_DEFS.get(cell.item)
        desc = f" — {idef.description}" if idef else ""
        parts.append(f"You see a {cell.item} here{desc}.")

    msg = " ".join(parts)
    world.log(msg)
    return msg


def _pick_up(world: World) -> str:
    ax, ay = world.agent_pos
    cell = world.cell_at(ax, ay)

    if not cell.item:
        return "There is nothing here to pick up."

    idef = ITEM_DEFS.get(cell.item)
    if idef and not idef.pickable:
        return f"The {cell.item} can't be picked up."

    item_name = cell.item
    cell.item = None
    world.inventory.append(item_name)
    world.step_count += 1

    # Track gem collection
    if item_name == "gem":
        world.gems_collected += 1
        remaining = world.count_items_on_map("gem")
        if remaining == 0:
            world.done = True
            world.success = True
            msg = (
                f"You pick up the gem ({world.gems_collected}/{world.total_gems}). "
                "🎉 All gems collected — task complete!"
            )
        else:
            msg = (
                f"You pick up the gem ({world.gems_collected}/{world.total_gems} collected, "
                f"{remaining} remaining on the map)."
            )
    else:
        msg = f"You pick up the {item_name}."

    world.log(msg)
    return msg


def _use_item(world: World, item: str, direction: str) -> str:
    if item not in world.inventory:
        return f"You don't have '{item}' in your inventory. Inventory: {world.inventory or '(empty)'}."

    ax, ay = world.agent_pos
    if direction == "here":
        tx, ty = ax, ay
    else:
        dx, dy = DIRECTIONS.get(direction, (0, 0))
        tx, ty = ax + dx, ay + dy

    if not world.in_bounds(tx, ty):
        return f"There's nothing to the {direction}."

    target = world.cell_at(tx, ty)

    # Key on locked door
    if item == "key" and target.terrain == Terrain.DOOR_LOCKED:
        target.terrain = Terrain.DOOR_OPEN
        world.inventory.remove("key")
        world.step_count += 1
        msg = f"You use the key on the locked door to the {direction}. It swings open!"
        world.log(msg)
        return msg

    world.step_count += 1
    return f"You try using the {item} {'' if direction == 'here' else 'to the ' + direction}, but nothing happens."


def _look(world: World) -> str:
    ax, ay = world.agent_pos
    lines = [f"You look around carefully from ({ax}, {ay})."]

    here = world.cell_at(ax, ay)
    if here.label:
        lines.append(f"Location: {here.label}.")
    if here.item:
        idef = ITEM_DEFS.get(here.item)
        desc = f" — {idef.description}" if idef else ""
        lines.append(f"At your feet: {here.item}{desc}.")

    lines.append("")
    for dir_name, (dx, dy) in DIRECTIONS.items():
        nx, ny = ax + dx, ay + dy
        if not world.in_bounds(nx, ny):
            lines.append(f"  {dir_name.upper():5s}: boundary")
            continue
        nc = world.cell_at(nx, ny)
        passable = world.is_passable(nx, ny)
        terrain = TERRAIN_NAMES[nc.terrain]
        entry = f"  {dir_name.upper():5s} ({nx},{ny}): {terrain}"
        if not passable:
            entry += " [BLOCKED]"
        if nc.item:
            idef = ITEM_DEFS.get(nc.item)
            entry += f", contains: {nc.item}" + (f" ({idef.description})" if idef else "")
        if nc.label:
            entry += f" [{nc.label}]"
        lines.append(entry)

    world.step_count += 1
    return "\n".join(lines)
