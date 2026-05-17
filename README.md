# LLM Agent in a Virtual World

A clean, modular harness that places a Claude LLM inside a grid-based virtual world where it can perceive, reason, and act.

Built as a solution to the [Humanoid](https://humanoidrobots.ai) software engineering intern challenge.

---

## Quick start

```bash
# 1. Clone / unzip
git clone <your-repo-url>
cd humanoid-agent

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the default scenario
python main.py

# Run all scenarios
python main.py --scenario all

# Save the episode log
python main.py --scenario key_door --save-log run.json
```

---

## Scenarios

| Name | Description | Map size | Difficulty |
|---|---|---|---|
| `key_door` | Find the key, unlock the door, reach the exit | 9×7 | Easy |
| `gem_collector` | Collect all 3 gems scattered across a larger map | 11×9 | Medium |
| `maze` | Navigate a hand-crafted maze to the exit | 13×11 | Hard |

---

## Architecture

```
humanoid-agent/
├── main.py               CLI entry point
├── requirements.txt
├── examples/
│   └── key_door_run.json   sample episode log
└── src/
    ├── world.py          Grid engine: World, Cell, Terrain, ItemDef
    ├── actions.py        Action space + execution (move, pick_up, use_item, look, declare_done)
    ├── observation.py    World → structured text observation
    ├── agent.py          LLM harness: Claude API ↔ World
    └── scenarios.py      Pre-built maps + scenario registry
```

### The agent harness (`agent.py`)

The harness is the interesting part.  Each step:

1. **Build observation** — the full world state is serialised into a structured text block (map + surroundings + inventory + event log)
2. **Call Claude** — observation is sent as the user message; available actions are passed as Claude *tools*
3. **Parse tool call** — Claude returns a `tool_use` block naming the action and its parameters
4. **Execute** — `execute(world, action_name, params)` mutates the world and returns a plain-English result
5. **Repeat** until `world.done` or `max_steps` reached

Each step is a **stateless API call** — no conversation history is kept between steps.  Instead, the world's `event_log` (last 8 events) is embedded in every observation, giving the agent short-term memory without needing conversation state.

### Observation design (`observation.py`)

The agent sees:

```
┌─── OBSERVATION ──────────────────────────────────────────┐
│  Step     : 3 / 40
│  Goal     : Find the key (k), unlock the locked door (D)…
├─── AGENT STATUS ─────────────────────────────────────────┤
│  Position : (1, 2)  [Storage room]
│  Inventory: ['key']
│  On ground: key — A small iron key — use it on locked door
├─── SURROUNDINGS ─────────────────────────────────────────┤
│  NORTH (1,1): floor
│  SOUTH (1,3): floor
│  EAST  (2,2): floor
│  WEST  (0,2): solid wall  [blocked]
├─── RECENT EVENTS ────────────────────────────────────────┤
│  • You move south to (1, 2). You see a key here.
├─── MAP  (@ = you) ───────────────────────────────────────┤
│  # # # # # # # # #
│  # . . . . . . . #
│  # @ k . . . . . #
│  # . . . . . . . #
│  # . . . . . . . #
│  # . . . . D . G #
│  # # # # # # # # #
│  Legend: @ you  . floor  # wall  D locked-door  _ open-door
│          G goal  k key  * gem  t torch  C crate
│  Coords: x increases going EAST, y increases going SOUTH
└──────────────────────────────────────────────────────────┘
```

Three information layers work together:
- **Map**: global spatial layout — the agent can plan paths
- **Surroundings**: local detail — what's immediately adjacent, what's passable
- **Event log**: short-term memory — what just happened, avoiding repeated mistakes

### Action space (`actions.py`)

Actions are defined as Claude tool schemas and executed by a simple dispatch function:

| Tool | Parameters | Effect |
|---|---|---|
| `move` | direction | Move one step; reports terrain, items, and goal detection |
| `pick_up` | — | Picks up item at agent's feet; tracks gem collection |
| `use_item` | item, direction | Uses inventory item on adjacent cell (key → door) |
| `look` | — | Detailed scan of current + 4 adjacent cells |
| `declare_done` | reason | Signals goal complete (for collection tasks) |

Using Claude's native tool-use API is intentional: it produces structured, typed action calls with no string parsing, and it's the same pattern used in production agentic systems.

---

## Example output

```
══════════════════════════════════════════════════════════════
  GOAL: Find the key (k), unlock the locked door (D), and step onto the goal tile (G).
══════════════════════════════════════════════════════════════
# # # # # # # # #
# @ . . . . . . #
# . k . . . . . #
# . . . . . . . #
# . . . . . . . #
# . . . . D . G #
# # # # # # # # #

── Step 1 ──
  [think] I need to find the key first. Looking south, I can see there's a key at (1,2)…
  [action] look({})
  [result] You look around carefully from (1, 1).
── Step 2 ──
  [action] move({"direction": "south"})
  [result] You move south to (1, 2). You see a key here.
── Step 3 ──
  [action] pick_up({})
  [result] You pick up the key.
  ...
── Step 11 ──
  [think] The door is to my east. I have the key. I'll use it now.
  [action] use_item({"item": "key", "direction": "east"})
  [result] You use the key on the locked door to the east. It swings open!
── Step 14 ──
  [action] move({"direction": "east"})
  [result] 🎉 You step onto the GOAL tile — task complete!

══════════════════════════════════════════════════════════════
  EPISODE COMPLETE — SUCCESS ✓
  Steps used: 14 / 40
══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────┐
│  RESULTS                                                │
├──────────────────┬───────────┬─────────────────────────┤
│  Scenario        │  Outcome  │  Steps                  │
├──────────────────┼───────────┼─────────────────────────┤
│  key_door        │  ✓ SUCCESS  │  14 / 40               │
└──────────────────┴───────────┴─────────────────────────┘
```

A full episode log is in [`examples/key_door_run.json`](examples/key_door_run.json).

---

## Design choices

**Tool use over free-text parsing**  
Using Claude's tool-call API gives structured, typed actions with zero prompt engineering for output format.  The same approach is used in production agent frameworks (LangChain, OpenAI Assistants, etc.) and maps naturally to how real robot action spaces work.

**Stateless observation over conversation history**  
Each step sends a single user message with the full world state encoded in it.  No conversation history is maintained.  This keeps API costs low and makes debugging easy — every step is self-contained and reproducible.  The world's `event_log` gives the agent the context it needs without requiring a growing message list.

**ASCII map in the observation**  
Grid-based environments are spatial.  Including the rendered map gives the LLM something to reason about geometrically — "the door is at the far right, the key is top-left, I need to collect the key first and then head right."  Token-efficient and surprisingly effective.

**Layered surroundings + map**  
The surroundings panel gives *precise* local information (is this cell passable? what item is here?) while the map gives *approximate* global context.  Together they let the agent plan at two scales.

**What didn't work (early iterations)**  
- Returning only JSON observations made the LLM less reliable — it would lose track of spatial position.  The ASCII map fixed this.
- Maintaining full conversation history added latency and cost without improving performance for environments where the observation is already complete.
- Allowing free-text action responses required fragile regex parsing.  Tool use eliminated this entirely.

---

## Extending the system

**New scenarios**: add a function to `src/scenarios.py` that builds and returns a `World`, then append a `Scenario` entry to the `SCENARIOS` list.

**New actions**: add a tool schema to `ACTION_TOOLS` in `src/actions.py` and a matching handler in `execute()`.

**New items**: add an `ItemDef` entry to `ITEM_DEFS` in `src/world.py` and handle the item in `_use_item` or `_pick_up` in `actions.py`.

**Different models**: pass `--model claude-opus-4-20250514` or any compatible model string to `main.py`.

---

## Requirements

- Python 3.10+
- `anthropic >= 0.25.0`
- An [Anthropic API key](https://console.anthropic.com/)
