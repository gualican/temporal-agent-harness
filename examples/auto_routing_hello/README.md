# Auto-routing hello-world OpenAI Agents agent

A copy of [`examples/openai_hello`](../openai_hello) with one change: the agent's model
starts as `"auto"` instead of a fixed model name. Every model call gets classified by
TypeSafe's **Jev** ([jev-model-router](https://github.com/gualican/jev-model-router)) and
routed to a model by tier, instead of always calling the same one.

## What it demonstrates

- **`JevAutoModelActivity`**
  (`temporal_agent_harness/ai_sdks/openai_agents/jev_auto_routing.py`) subclasses
  `ModelActivity`, intercepting `model_name == "auto"` requests to classify the latest user
  message and resolve a real model name before delegating to the base implementation.
  Requests with any other `model_name` are unaffected — this is purely additive.
- **Manual activity registration.** `OpenAIAgentsPlugin` has no way to inject a custom
  `ModelActivity`, so `worker.py` sets `register_activities=False` on the plugin and
  registers the `JevAutoModelActivity` instance's two methods on the `Worker` itself. See
  the module docstring in `jev_auto_routing.py` for why.
- **`/model` at runtime.** `workflow.py` wires the harness's existing (generic, opt-in)
  `slash_commands.model_selector(...)` with `"auto"` added to its choices — the same
  pattern `examples/monty/conversational_workflow.py` already uses for its own fixed model
  list. Send `/model auto` or `/model gpt-5.1` mid-session to switch.

## Layout

| File | Role |
|---|---|
| `workflow.py` | `AutoRoutingHelloAgentWorkflow` — same as `OpenAIHelloAgent`, but `model=AUTO_MODEL` and a `/model` slash command. |
| `worker.py` | Registers `JevAutoModelActivity` instead of letting the plugin build its own `ModelActivity`. |
| `agents.toml` | Registry entry that makes this agent selectable in the shared web UI. |

## Run it

Prereq: from the repo root, `cp .env.example .env.local` and set `OPENAI_API_KEY` (the
routed models are OpenAI models), `TYPESAFE_API_KEY` (Jev classifies each request), and your
Temporal connection profile. Then, each in its own terminal:

```sh
just temporal          # 1. local Temporal dev server (or bring your own)
just session-manager   # 2. packaged session-manager worker
just server            # 3. builds the Svelte UI, then serves API + UI on http://localhost:8000
just worker            # 4. the agent worker
```

Open http://localhost:8000, pick **Auto-Routing Hello**, and chat. Try a simple message
("hi") vs. a harder one, and watch the worker's stdout — each call logs which tier Jev
picked. Send `/model gpt-5.1` to pin one model, then `/model auto` to go back to
per-call routing.

Without `just`, the equivalent commands (from the repo root):

```sh
uv run --group examples python -m examples.session_manager_worker
uv run --group examples python -m examples.app examples/auto_routing_hello/agents.toml --host 0.0.0.0 --port 8000
uv run --group examples python -m examples.auto_routing_hello.worker
```
