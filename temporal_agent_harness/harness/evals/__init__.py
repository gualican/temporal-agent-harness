# ABOUTME: Evals for harness agents: grade one agent turn with deterministic checks plus an LLM
# judge, in three modes (online / side_effect / offline). Import from the submodules — this
# __init__ stays empty because ``workflow`` is loaded inside the workflow sandbox and must not drag
# in ``client`` (which holds a Temporal client).
#
#   models   EvalSpec (what "good" means for an agent), AgentRunResult (what happened),
#            EvalRequest / EvalReport, and the check/judge result types
#   checks   run_checks(run, spec) — pure, deterministic
#   workflow EvalWorkflow — checks + judge, registered on the worker (needs the OpenAI Agents plugin)
#   helpers  run_turn_eval(...) — workflow side: attach an online or side-effect eval to a turn
#   client   evaluate_run(...) / format_report(...) — client side: grade a past run offline
#
# The spec and the agent's instructions travel IN the request, so nothing here depends on how you
# store agent definitions.
