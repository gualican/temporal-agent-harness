# Auto model routing via jev-model-router

> Status: implemented (`temporal_agent_harness/ai_sdks/openai_agents/jev_auto_routing.py`,
> `examples/auto_routing_hello`). This note records why it's shaped the way it is, for
> anyone extending it or wiring the same idea into another `ai_sdks` integration.

The goal: an agent whose model isn't fixed — each call is classified (by TypeSafe's Jev, via
[jev-model-router](https://github.com/gualican/jev-model-router)) and routed to a model by
tier, instead of the app author picking one model for everything up front.

## Why not a custom `ModelProvider`

That's the SDK's own extension point for "resolve a model name to a `Model`," so it looks
like the obvious place. It isn't: `ModelProvider.get_model(model_name: str | None) -> Model`
receives only the name string — never the conversation. A router needs to see what's being
asked to decide anything, so a `ModelProvider` alone can't make a content-based decision.

Routing has to happen one layer up, inside the Temporal *activity* that resolves the model
(`ModelActivity.invoke_model_activity[_streaming]` in `_invoke_model_activity.py`), which is
the one place `model_name` and the actual conversation (`input["input"]`) are both in scope
together. This is also architecturally the right place for it regardless: activities are
Temporal's designated home for non-deterministic IO, and an extra classifier call is exactly
that.

## Why a `ModelActivity` subclass, not a plugin change

`ModelActivity`'s own docstring says it's "a class wrapper... to allow model customization,"
and subclassing it does work correctly at the `temporalio` SDK level (`@activity.defn` keys
off the function object created at each `def`, not the defining class). The catch:
`OpenAIAgentsPlugin` always constructs the base class itself when `register_activities=True`
(the default) — there's no parameter to hand it a custom instance. Rather than edit the
plugin (a core, upstream-tracked file), this uses the escape hatch the plugin already ships
for exactly this situation: `register_activities=False`, then register the subclass
instance's two methods on the `Worker` directly. Net result: the feature is entirely new
files — nothing in the harness's own tracked code changes, which matters for keeping this
fork rebaseable against upstream.

## What jev-model-router contributes vs. what this glue owns

Only `jev_model_router.classifier.classify()` (the Jev call itself) and
`jev_model_router.router.resolve_tier()` (the confidence-gating / stakes-override policy) are
reused — not `route()`, which also calls Anthropic directly. `jev_model_router.config.TIER_TO_MODEL`
is Claude-specific; this harness is multi-provider, so `JevAutoModelActivity` takes its own
`tier_to_model` from the app instead.

## Known limitation: no cross-call caching within a turn

A single user turn can trigger several `invoke_model_activity` calls as the agent works
through tool calls. Each one re-runs classification independently — the extracted user text
is unchanged within a turn, so the result is almost always identical, but it's still a
redundant Jev call per sub-call. Given Jev's low cost/latency (see jev-model-router's own
benchmark: median ~400ms, ~$0.042/MTok input, output free) this seemed like an acceptable v1
simplification rather than threading a cached decision through workflow-level state. Worth
revisiting if a workload does many tool-loop steps per turn.

## If this pattern is wanted for another `ai_sdks` integration

`google_genai_plugin`'s model-invocation activities are a transparent HTTP relay with no
"resolve model name to a `Model` instance" step to hook, and `pydantic_ai_harness.py` uses
the upstream `pydantic_ai.durable_exec.temporal` plugin unmodified — neither has the same
seam this relies on. Porting the idea there would mean finding (or adding) an analogous hook
in each integration's own model-resolution path, not reusing this module directly.
