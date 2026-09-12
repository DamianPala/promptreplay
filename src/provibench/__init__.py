"""provibench: record real agent-harness API traffic and replay it against LLM providers.

`core/` holds the mechanics the CLI Design Standard requires, `commands/` the command
groups, and `bench/` the domain library (trace format, proxy, replay engine, pricing) with
no dependency on `core/` or `commands/`.
"""
