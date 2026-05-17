"""
observation.py — World-state → LLM observation.

This is the *representation* layer: it decides what the agent
knows and how that information is structured.

Design choices:
  - Structured ASCII panels keep spatial reasoning grounded.
  - The map gives global layout; surroundings give local detail.
  - Inventory and goal are always visible — no hidden state.
  - Recent events let the agent track progress without full history.
"""

from __future__ import annotations

from .actions import DIRECTIONS
from .world import ITEM_DEFS, TERRAIN_NAMES, Terrain, World


def build_observation(world: World) -> str:
    """
    Construct the full text observation from the current world state.
    This is exactly what the LLM receives on each step.
    """
    ax, ay = world.agent_pos
    here = world.cell_at(ax, ay)

    # ── Header ────────────────────────────────────────────────────────────────
    lines = [
        "┌─── OBSERVATION ──────────────────────────────────────────┐",
        f"│  Step     : {world.step_count} / {world.max_steps}",
        f"│  Goal     : {world.goal}",
        "├─── AGENT STATUS ─────────────────────────────────────────┤",
        f"│  Position : ({ax}, {ay})"
        + (f"  [{here.label}]" if here.label else ""),
        f"│  Inventory: {world.inventory or '(empty)'}",
    ]

    # Gem progress
    if world.total_gems > 0:
        remaining = world.count_items_on_map("gem")
        lines.append(
            f"│  Gems     : {world.gems_collected}/{world.total_gems} collected, "
            f"{remaining} still on map"
        )

    # Item underfoot
    if here.item:
        idef = ITEM_DEFS.get(here.item)
        desc = f" — {idef.description}" if idef else ""
        lines.append(f"│  On ground: {here.item}{desc}")

    # ── Surroundings ──────────────────────────────────────────────────────────
    lines.append("├─── SURROUNDINGS ─────────────────────────────────────────┤")

    for dir_name, (dx, dy) in DIRECTIONS.items():
        nx, ny = ax + dx, ay + dy
        if not world.in_bounds(nx, ny):
            lines.append(f"│  {dir_name.upper():5s}: [boundary]")
            continue

        nc = world.cell_at(nx, ny)
        passable = world.is_passable(nx, ny)
        terrain = TERRAIN_NAMES[nc.terrain]

        entry = f"│  {dir_name.upper():5s} ({nx},{ny}): {terrain}"
        if not passable:
            if nc.terrain == Terrain.DOOR_LOCKED:
                entry += "  ← USE KEY HERE"
            else:
                entry += "  [blocked]"
        if nc.item:
            idef = ITEM_DEFS.get(nc.item)
            entry += f" | item: {nc.item}" + (f" ({idef.description})" if idef else "")
        if nc.label:
            entry += f" [{nc.label}]"
        lines.append(entry)

    # ── Recent events ─────────────────────────────────────────────────────────
    lines.append("├─── RECENT EVENTS ────────────────────────────────────────┤")
    events = world.event_log[-5:] if world.event_log else ["(none yet)"]
    for e in events:
        lines.append(f"│  • {e}")

    # ── ASCII map ─────────────────────────────────────────────────────────────
    lines.append("├─── MAP  (@ = you) ───────────────────────────────────────┤")
    for row in world.render().split("\n"):
        lines.append(f"│  {row}")
    lines.append(
        "│  Legend: @ you  . floor  # wall  D locked-door  _ open-door"
    )
    lines.append(
        "│          G goal  k key  * gem  t torch  C crate"
    )
    lines.append(
        "│  Coords: x increases going EAST, y increases going SOUTH"
    )
    lines.append("└──────────────────────────────────────────────────────────┘")

    return "\n".join(lines)
