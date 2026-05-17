"""
agent.py — LLM agent harness.

This is the heart of the system: it wires the Claude API to the
world via the tool-use interface.

Architecture
------------
  Observation (text)
       ↓
  Claude API  (system prompt + tools)
       ↓
  Tool-use response  →  action_name + params
       ↓
  execute(world, action_name, params)
       ↓
  Result message (fed back as tool_result)

Each step is a *stateless* API call: the full world state is encoded
in the observation, so the agent doesn't need conversation history to
reason about the environment.  The event_log inside the world acts as
the agent's short-term memory.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import anthropic

from .actions import ACTION_TOOLS, execute
from .observation import build_observation
from .world import World

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous agent navigating a grid-based virtual world.
Your job is to accomplish the stated goal by calling the available tools.

== World conventions ==
- Grid coordinates: x increases going EAST, y increases going SOUTH.
- @ = your position  . = floor  # = wall
- D = locked door (need key to open)  _ = open door  G = GOAL tile
- k = key  * = gem  t = torch  C = crate

== Available actions ==
- move(direction)         — move one step (north/south/east/west)
- pick_up()               — pick up item at your current position
- use_item(item, direction) — use inventory item on adjacent cell (e.g. key on north door)
- look()                  — detailed scan of current + adjacent cells
- declare_done(reason)    — signal goal complete (for collection goals)

== Strategy ==
1. Read the observation carefully — map, surroundings, inventory, recent events.
2. Plan: what do you need, where is it, what's in the way?
3. If a door is locked (D), find and pick up a key (k) first, then use key on the door.
4. For navigation goals, step onto the G tile to win.
5. For collection goals (gems), pick up every item then call declare_done.
6. Be systematic — explore methodically, don't backtrack needlessly.
7. You have limited steps, so be efficient.

Think briefly before each action, then call exactly one tool."""

# ── Agent ─────────────────────────────────────────────────────────────────────


class LLMAgent:
    """
    Connects the Claude API to a World via tool use.

    Each call to step() is a complete round-trip:
      build observation → Claude → parse tool call → execute → return result
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = True,
    ) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.verbose = verbose

    # ── Single step ───────────────────────────────────────────────────────────

    def step(self, world: World) -> Tuple[str, str]:
        """
        Run one agent step.

        Returns:
            (action_name, result_message)
        """
        observation = build_observation(world)

        # ── Call Claude with tool use ──
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=ACTION_TOOLS,
            messages=[{"role": "user", "content": observation}],
        )

        # ── Parse response ────────────────────────────────────────────────────
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        text_block = next(
            (b for b in response.content if b.type == "text"), None
        )

        reasoning = text_block.text.strip() if text_block else ""

        if tool_block is None:
            # Shouldn't happen with tool_choice not forced, but handle gracefully
            if self.verbose:
                print("[Agent] No tool call returned — skipping step.")
            return "none", "No action taken."

        action_name = tool_block.name
        params: Dict[str, Any] = tool_block.input

        # ── Log to console ────────────────────────────────────────────────────
        if self.verbose:
            if reasoning:
                # Show first 180 chars of reasoning
                short = reasoning[:180] + ("…" if len(reasoning) > 180 else "")
                print(f"  [think] {short}")
            param_str = json.dumps(params) if params else ""
            print(f"  [action] {action_name}({param_str})")

        # ── Execute ───────────────────────────────────────────────────────────
        result = execute(world, action_name, params)

        if self.verbose:
            print(f"  [result] {result}")

        return action_name, result

    # ── Full episode ──────────────────────────────────────────────────────────

    def run_episode(self, world: World) -> Dict[str, Any]:
        """
        Run until the world reports done or max_steps is reached.

        Returns a summary dict suitable for logging / JSON output.
        """
        episode_log: List[Dict[str, Any]] = []

        if self.verbose:
            print()
            print("═" * 62)
            print(f"  GOAL: {world.goal}")
            print("═" * 62)
            print(world.render())
            print()

        while not world.done and world.step_count < world.max_steps:
            if self.verbose:
                print(f"── Step {world.step_count + 1} ──")

            action, result = self.step(world)

            episode_log.append(
                {
                    "step": world.step_count,
                    "action": action,
                    "result": result,
                    "agent_pos": list(world.agent_pos),
                    "inventory": list(world.inventory),
                }
            )

        # ── Timed out ─────────────────────────────────────────────────────────
        if not world.done:
            world.done = True
            world.success = False

        if self.verbose:
            outcome = "SUCCESS ✓" if world.success else "FAILED  ✗"
            print()
            print("═" * 62)
            print(f"  EPISODE COMPLETE — {outcome}")
            print(f"  Steps used: {world.step_count} / {world.max_steps}")
            print("═" * 62)
            print(world.render())

        return {
            "goal": world.goal,
            "success": world.success,
            "steps": world.step_count,
            "max_steps": world.max_steps,
            "log": episode_log,
        }
