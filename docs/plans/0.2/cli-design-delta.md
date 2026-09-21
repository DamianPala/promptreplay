# CLI Design Standard 0.1.0 delta

Audited against the standard text in `cli-design-standard-0.1.0.md`, not the draft that
generated the original scaffold. The evidence is the runtime parser, `promptreplay schema`,
both `--json` positions, command help, error handling, confirmation behavior, `config show`,
`completion`, the registered command implementations, and the shipped documentation.

The conformance claim remains `0.1.0-draft.7`. Several standard MUST requirements are
open, so changing the claim to `0.1.0` would be false.

There are 127 applicable rows: 78 `ok`, 24 `fixed`, and 25 `open`. The B1 greenfield rule
means the B3 and B4 brownfield exceptions do not apply to promptreplay's own command paths,
flags, defaults, or result names. The B2 route, claiming `0.1.0` with `conforming: false` per
command, was considered and rejected: O6b alone would exclude nine commands, and R1b and R2b
would exclude the rest, leaving a claim covering `config show` alone. An honest draft claim
carries more information. B1 classifies the interface parts, B2 defines command coverage,
B3 lists retained names and defaults, and B4 covers retained meanings with replacements.

| ID | Status | Evidence, change, or remaining gap |
|---|---|---|
| D1 | fixed | README, design docs, schema, help, completion, and the agent skill now describe the same command set; `tests/test_skill.py` checks the skill's fenced commands against the parser. |
| D2 | ok | Root help, `schema`, and the skill form the three discovery layers. |
| D3 | fixed | Root help is a standalone catalog and points to `promptreplay schema` and `--json`; command help is generated from the same parser. |
| D3a | fixed | `app.py` epilog now includes the introspection literal, JSON guidance, and the `NO_INPUT` prompt rule. |
| D3b | fixed | Added help text to the six positional arguments whose empty descriptors suppressed Click's `Positional arguments:` section; every command now describes purpose, usage, arguments, flags, and defaults. |
| D4 | ok | Names and shared flags use a predictable command tree and consistent meanings. |
| D4a | ok | `config show` is the only nested operation; the remaining commands are direct verbs or nouns with one operation. |
| D4b | open | `inspect`, `endpoints`, and `prices` are greenfield names instead of the recommended `get`/`list` vocabulary. Recommendation: choose the preferred names before the first release; this SHOULD drift does not block the current claim. |
| D4c | ok | `--json`, `--yes`, `--force`, `--budget`, `--warm`, and `--timeout` keep one meaning wherever used. |
| D4d | ok | Root `--version` exists and prints the package version without a `v` prefix. |
| D5 | ok | `schema` emits the machine-readable interface. |
| D5a | ok | `schema` is a root-reserved command and cannot dispatch a domain operation. |
| D5b | fixed | Introspection remains derived from Click; the timeout descriptor was aligned with the runtime by adding standard duration syntax while preserving decimal seconds. Its default changed from `300.0` to `"300s"`, which needs no `schema_version` bump because I1a declares `default` may be a string, number, boolean, or array, and D8a governs introspection fields rather than flag-default values. |
| D5c | ok | Schema uses no target loading, authentication, network, prompt, or trace/run state and writes only JSON to stdout. |
| D5d | ok | Bare `schema` returns the index and a space-separated command path returns detail. |
| D5e | ok | Unknown paths become `invalid_input` with exit `2` and a help/index hint. |
| D6 | ok | The index has the required stable selection fields. |
| D6a | ok | `schema_version`, `tool_version`, `conformance`, `global_flags`, `format_defaults`, `exit_codes`, and sorted `commands` are present. |
| D6b | ok | Entries contain only unique full names, descriptions, and R1 effects. |
| D6c | open | The object is honest about the retained draft claim, but cannot claim `0.1.0` while MUST rows below remain open. |
| D7 | fixed | Detail exposes parser inputs, effects, confirmation, interactivity, output schemas, and conditional `output_description` for completion, report, probe, and sweep. |
| D8 | ok | The schema version is the positive decimal string `"1"`; unknown optional fields remain forward-compatible. |
| D8a | ok | The added optional output-description metadata does not require a schema-version bump. |
| D8b | ok | No compatible tool-version range is currently promised; the current command paths, defaults, error kinds, and output fields are tested together. |
| D9 | open | A partial `probe`, `sweep`, or `replay` names its one clear follow-up, `report RUN_DIR`, as the `operation_failed` error's `hint` rather than a structured `next` breadcrumb; D9 is a SHOULD, and the run's location is already in the error's `context` for a script to build its own command from. Recommendation: add `next` in a later output-contract change. |
| D9a | ok | No result or F3 error currently emits `next`, so no malformed argv breadcrumb is exposed. |
| D9b | open | The partial-run follow-up stays a `hint`, not the standard `Next:` label; unchanged by the O5a fix, which only moved the run's own data (the result document) from `error.context` to stdout and left `context` naming just the run's location. Recommendation: render the same breadcrumb when D9 is implemented. |
| D9c | ok | Results without a natural continuation omit `next`. |
| D10 | fixed | `skills/promptreplay/SKILL.md` adds domain context and points agents to `schema` instead of copying the command catalog. |
| I1 | fixed | All accepted arguments and flags are generated into descriptors, and a schema test now rejects any empty descriptor name, description, or type. |
| I1a | fixed | Added descriptions to the six positional arguments; descriptor names, descriptions, types, required/default values, choices, aliases, repeatability, and stdin markers match parser metadata. |
| I1b | ok | Click boolean flags do not consume the following positional token. |
| I1c | ok | Required values omit defaults; built-in defaults are typed; runtime-resolved paths are described rather than emitted as parser defaults. |
| I1d | ok | Descriptions state duration syntax, path/config resolution, bounds, and global/command relationships where needed. |
| I2 | ok | Configuration is declared once and resolved deterministically. |
| I2a | ok | README and help document config files, target files, declared `PROMPTREPLAY_*` variables, API-key environment names in `targets.toml`, and `NO_INPUT`. |
| I2b | ok | Resolution is flag, environment, configuration file, then built-in default, and the order is documented. |
| I2c | ok | `config show` returns every resolved setting and source; path values are resolved and secrets are masked. |
| I3 | open | Trace documents are accepted through a path only (a `--replace OLD=NEW` literal is short text, not a document; see I7a for its buffering). Recommendation: add the `-` form for document-sized inputs without changing existing forms. |
| I3a | open | `inspect`, `probe`, `sweep`, `replay`, and `scrub` accept trace documents by path but do not accept `-`. Recommendation: add file and stdin forms for document-sized inputs. |
| I4 | ok | API keys are selected by environment-variable names in targets configuration, never passed as CLI values. |
| I4a | ok | Native and OpenRouter targets have non-interactive environment sources for credentials. |
| I4b | ok | Keys are not part of help, schema, diagnostics, or normal output. |
| I5 | ok | Prompting depends on TTY state, selected JSON output, stdin claims, and `NO_INPUT`; missing consent fails closed. |
| I5a | ok | Confirmation prompts use TTY stderr/stdin only; `--json`, redirected stdin, and `NO_INPUT` require explicit `--yes`. |
| I6 | ok | Click and the command layer reject unknown, conflicting, and invalid input as `invalid_input` before domain effects. |
| I6a | ok | Usage failures use F3 `invalid_input` and exit `2`. |
| I6b | ok | Click honors `--` and accepts global flags before or after command paths. |
| I6c | ok | Locally checkable arguments, target specs, paths, and preconditions are checked before writes or requests. |
| I6d | ok | Errors identify the bad input and usually provide accepted syntax or a nearest command hint. |
| I7 | open | Trace parsing and other caller-controlled document reads have no declared finite byte limit. Recommendation: add a bounded document reader and describe its limit in the input descriptors. |
| I7a | open | Trace files and scrub replacement strings can be buffered without a maximum before side effects. |
| I8 | fixed | Network calls retain finite defaults, and `record` now accepts the standard timeout option for its otherwise unbounded wait. |
| I8a | ok | Endpoint, price-table, pre-check, probe, and replay network clients use finite defaults; probe/replay use a finite `300s` default. |
| I8b | fixed | `record --timeout DURATION` stops the proxy after the requested duration; its descriptor says the wait is unbounded by default, so the no-flag behavior remains waiting for Ctrl-C. |
| I8c | ok | No other unbounded mode is silently selected; the long-running recorder is explicitly documented. |
| R1 | open | The global effect declarations are present, and `report --output-file` now replaces its destination (a report is derived from the run alone, so the repeat writes the same bytes), but `completion --install` and scrub's OUT product can still fail on an immediate repeat unless `--force` is added. |
| R1a | ok | Every index entry and detail declares one allowed effects value. |
| R1b | open | `completion` and `scrub` are declared `idempotent` while their existing-destination precondition makes a successful repeat fail (`report` no longer has that gate). Recommendation: either make repeat success unconditional or classify those calls/commands as non-idempotent. |
| R2 | open | Narrow writes and API-credit spending are identified, but existing overwrite and recording calls do not all carry the safeguards implied by irreversibility. |
| R2a | ok | The commands name their trace, run, report, or completion target; no command claims a wide mutation. |
| R2b | open | `record` and forced destination overwrites have no confirmation gate. Recommendation: add confirmation or document a restore operation before claiming full release conformance. |
| R3 | open | The probe/replay/sweep gate is complete, but `record` can change a trace without the same gate. |
| R3a | open | `record` has no `--yes` despite writing a new or appended trace. |
| R3b | ok | Gated probe, replay, and sweep calls resolve estimates/checks first and fail with `confirmation_required` before API requests when consent is unavailable. |
| R3c | fixed | `probe`, `sweep`, and `replay` accept `--dry-run` and never require `--yes` under it; `--yes` alongside it is accepted and ignored, verified by re-checking `requires_confirmation` and the mocked transport's zero requests with and without `--yes`. |
| R3d | ok | `--yes` confirms consent and `--force` only overrides existing-destination/precondition checks. |
| R4 | fixed | `probe`, `sweep`, and `replay` spend real API credit without the caller naming every affected target in advance, so a `--dry-run` preview under R4 applies the same way a wide mutation's would; all three now provide it. |
| R4a | fixed | `probe`, `sweep`, and `replay` provide `--dry-run`. |
| R4b | fixed | A `--dry-run` call leaves intended state unchanged (no request is sent), never fails with `confirmation_required`, and still fails for any other reason the real call would, `--budget` included, since `check_budget` runs before the dry-run branch. |
| R4c | fixed | A `--dry-run` success conforms to the same D7 `output` (the schema now declares its fields as optional, present only under `--dry-run`, per R4c's own rule for a value only the mutation produces), returns `changed: false`, and returns `requires_confirmation` true exactly when the same call without `--dry-run` and without `--yes` would gate in a non-interactive context. `probe`/`sweep` also report the pre-check plan and `--top` upper bound priced into the estimate; `targets` (R4c's own field name) is not applicable here since these commands do not name pre-existing targets, so the estimate's own `estimate` array serves the equivalent role. |
| R5 | ok | Mutating commands return structured results and report their state transition. |
| R5a | ok | All non-read-only schemas require boolean `changed`, and successful calls populate it. |
| R5b | ok | Note: a local `Path.exists()` check can race a concurrent writer, but no authoritative state reports a conflict here, so R5b does not apply to that observation. |
| R5c | open | Probe/replay POST retries have no idempotency key or equivalent duplicate-effect protocol. Recommendation: pass a provider-supported key where available, or document why measurement requests are safe to repeat. |
| R6 | open | Probe retries transient POSTs and the thinking fallback on `non_idempotent` measurement calls without an idempotency guarantee. Recommendation: bound and protect the retry protocol before release conformance. |
| O1 | ok | TTY detection for stdin, stdout, and stderr is kept independent. |
| O2 | fixed | Report uses `--format text|md|html` for rendering and `--output-file PATH` for a result destination; scrub uses positional OUT for its trace product and the same result-destination flag. |
| O2a | ok | The index declares `text`/`json`; completion's differing text defaults are emitted only in its detail. |
| O2b | ok | Every command accepts global `--json`; it selects JSON for documents and the result stays off diagnostics. `report`, the one command with `--format`, accepts `--format json` as the same selection, as its `format_defaults` index implies. |
| O2c | ok | Documents default to text on TTY and JSON off TTY; completion uses its declared native text format. |
| O2d | ok | Explicit JSON wins over detected defaults and JSON/plain conflicts are rejected. |
| O2e | fixed | The command now follows the clause, “a command MUST NOT accept `--output`; if it accepts a destination path, the flag MUST be named `--output-file`.” Report's `--output-file PATH` writes exactly the selected rendering or JSON result and leaves stdout empty. Scrub's OUT is its positional scrubbed-trace product, outside O2e's flag rule, while its result summary can also go to `--output-file`. |
| O3 | ok | Result data is separated from stderr diagnostics and progress. |
| O3a | ok | stdout carries result data; progress, estimates, warnings, prompts, and errors use stderr. |
| O3b | ok | Machine output has no terminal decoration; non-TTY diagnostics are complete lines. |
| O3c | ok | Human renderers use the same documents and report bounded/omitted values through labels or notes. |
| O3d | ok | JSON serialization is machine-safe and human renderers escape terminal control sequences. |
| O4 | ok | Registered document commands publish restricted JSON schemas. |
| O4a | ok | Schemas use only the standard's allowed subset. |
| O4b | ok | Every non-empty schema has an allowed type; nullable values use the two-type form. |
| O4c | ok | Enumerations describe only current parser/result choices. |
| O4d | ok | Objects declare properties and required fields; the schema tests enforce required subset of properties. |
| O5 | fixed | Every success document matches its schema, `probe`, `sweep`, and `replay` included, whose `output` now declares a required `partial`. |
| O5a | fixed | `probe`, `sweep`, and `replay` declare `partial` as required and always emit the result document on stdout, partial run included; a partial run's `operation_failed` error on stderr carries only `run_dir` (and `run_hex` for `probe`/`sweep`) in `context`, not the document again. Guarded by `tests/test_probe.py::test_probe_partial_failure_gives_json_callers_the_summaries` and its replay/sweep counterparts. |
| O5b | ok | JSON values preserve declared types and the writer does not silently truncate them. |
| O5d | fixed | Persisted timestamps use RFC 3339; the TTL result field is now `ttl[].offset_s`, and numeric durations carry units such as `_ms`, `_s`, or `_age_s`. Loaders accept legacy `offset` in an old result document. |
| O6 | open | Several successful documents contain potentially unbounded arrays. |
| O6b | open | `inspect`, `endpoints`, `history`, `compare`, `prices`, `report`, `probe`, `sweep`, and `record` have no finite result window/default `--limit`. Recommendation: add bounded collection envelopes and limits per command. |
| O6c | open | Those collections return bare arrays or command-specific objects without the required `items`/`has_more` page shape. |
| O8 | ok | Broken-pipe handling silences the output stream and exits successfully without a traceback. |
| F1 | open | Most exit meanings are stable, but replay records failed requests in a successful run and `record` treats interruption as success. |
| F1a | ok | Index and runtime map `0` to success, `1` to failure, and `2` to usage error. |
| F1c | fixed | A replay with a failed turn now exits non-zero: `partial: true` reaches stdout as an O5a result and an `operation_failed` error follows on stderr. Guarded by `tests/test_replay.py::test_replay_exits_non_zero_on_a_failed_turn`. |
| F1d | ok | Fine-grained failure reasons are represented by F3 kinds rather than extra exit codes. |
| F1e | ok | No command-specific exit code refinements are declared. |
| F2 | ok | Machine-readable failures have one F3 object on stderr, never stdout, and it is the last non-empty stderr line. |
| F2a | ok | Human diagnostics include the message and hint; machine diagnostics include the same fields in the object. |
| F2b | ok | JSON, non-TTY stderr, quiet mode, and early `--json` all require the object. |
| F2c | ok | The error object is emitted exactly once at the required location. |
| F3 | ok | Errors use one stable top-level `error` envelope. |
| F3a | ok | The envelope contains `kind`, `message`, and only the optional standard recovery fields used by the tool. |
| F3b | ok | `retryable`, `hint`, `action`, and `context` are emitted only when known; unsafe retries are not marked retryable. |
| F3c | ok | Used kinds retain their standard meanings, including `invalid_input`, `confirmation_required`, `operation_failed`, and `precondition_failed`. |
| F4 | open | Fallbacks are logged, but the selected packaged target fallback is not identified in a structured success result. |
| F4a | open | A timed-out replay POST can have an unobserved provider-side effect but is stored as a normal failed turn and the command can still exit `0`; recommendation: propagate `outcome_unknown`. |
| F4b | ok | Partial probe, sweep, and replay data now survives on stdout as the O5a result document rather than in `error.context`; the hint still identifies the persisted run. |
| F4c | open | Missing configured targets silently continue with packaged defaults after a stderr note; structured probe/replay results do not carry the substitution. Recommendation: fail or record the selected source in the result. |
| F5 | open | `record` suppresses `KeyboardInterrupt` and returns a success summary after Ctrl-C, contrary to the standard interruption result. Recommendation: distinguish intentional recorder stop from an external interrupt or document an explicit stop protocol. |
| H1 | fixed | Destination and rendering concepts now use canonical `--output-file PATH` and `--format NAME` names; all command flags remain kebab-case. |
| H1a | ok | The global and command-specific flag names are kebab-case and aliases appear in descriptors. |
| H1b | fixed | Report uses `--format NAME` and `--output-file PATH`; scrub's trace product is positional OUT and its result document uses `--output-file`; `--timeout` uses the required duration spelling. |
| H2 | ok | `completion` generates bash, zsh, and fish candidates from the installed parser and can install them idempotently. |
| H4 | ok | Color is controlled by `--color auto|always|never` and respects TTY, `TERM=dumb`, and `NO_COLOR`. |
| H4a | ok | ANSI decoration is omitted from non-TTY streams and machine-readable output. |
| H4b | ok | The flag name and values are exactly the standard's color contract. |
| H5 | open | Collection commands do not offer `--plain`; recommendation: add it when the O6 pagination contract is implemented. |

## Not applicable

These clauses have no current subject in promptreplay, so they have no delta row: `I3b` (no
command currently accepts stdin), `I5b`, `I5c`, and `I5d` (no editor, external approval, or
pager workflow), `I7b` (no command declares a path root), `R2c` (the `managed` extension is not
claimed), `R3e` (no command has declared a wide mutation that would take `--expect-targets`),
`R7`, `R7a`, `R7b`, `R7c`, and `R7d` (no work accepted for observation after the
initiating command), `O2f` (no delegating command), `O5c` (no capped single inline value), `O6a`
(no preview target list), `O7`, `O7a`, `O7b`, `O7c`, and `O7d` (no record-stream command),
`F1b` (no yes/no command), `H3` (no editor or pager), and `H5a`/`H5b` (the tool does not yet
support `--plain`).
