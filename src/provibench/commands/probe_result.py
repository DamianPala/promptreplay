"""Shaping a finished probe run's result: ordering, notes, progress lines, the partial exit.

Kept apart from `commands.probe` purely for its line budget: everything here runs after the
requests are sent, while `probe_phases` is everything that runs before them, so the two
modules split on that one seam rather than on any functional boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from provibench.commands.summary_view import message_lines, probe_report_text
from provibench.core.documents import Document
from provibench.core.errors import OperationFailed

if TYPE_CHECKING:
    from provibench.bench.probe import ProbeResult, ProbeRun
    from provibench.bench.probe_summary import ProbeSummary
    from provibench.bench.targets import RunSpec
    from provibench.core.context import Invocation

_ERROR_TRUNCATE = 120


def parallel_note(parallel: int) -> list[str]:
    """One note for a run that probed several endpoints at once, and none for a sequential one.

    A `run.json` reader comparing latency medians between runs has to know which of them
    were measured with other endpoints in flight on the same connection pool; the note is
    part of the run's record, not of its options, because it describes how the numbers were
    taken rather than what the protocol did.
    """
    if parallel <= 1:
        return []
    return [f"parallel {parallel}: latency measured with endpoints in flight together"]


def by_price(summaries: Sequence[ProbeSummary]) -> list[ProbeSummary]:
    """The summaries as a sweep reads them: cheapest effective prompt token first.

    That is the question a sweep asks — which endpoint to use this week — so the effective
    price leads and the hit rate breaks its ties, being the other half of why one endpoint
    is cheaper than another. An endpoint with no listed price cannot be ranked and sorts last.
    """
    return sorted(
        summaries,
        key=lambda s: (s.eff_per_m_prompt is None, s.eff_per_m_prompt or 0.0, -(s.hit_rate or 0.0)),
    )


def in_order(specs: Sequence[RunSpec], summaries: Sequence[ProbeSummary]) -> list[RunSpec]:
    """The endpoints in the order their summaries came out, for the run directory."""
    by_label = {spec.label: spec for spec in specs}
    return [by_label[summary.label] for summary in summaries]


def fail_on_partial(
    invocation: Invocation,
    run: ProbeRun,
    summaries: Sequence[ProbeSummary],
    document: Document,
    run_dir: Path,
) -> None:
    """Set `partial` and, when anything failed or was skipped, fail after the run is written.

    O5a: the result document is emitted on stdout exactly as a clean run's would be, partial
    or not, so an agent never has to unwrap it from `error.context`; a terminal instead gets
    the two tables on stderr before the error, which stays this command's own record and
    names only the run rather than repeating the document.
    """
    from provibench.bench.probe import is_failed
    from provibench.core.output import write_document

    failed = sum(1 for records in run.records.values() for record in records if is_failed(record))
    skipped = sum(summary.skipped for summary in summaries)
    document["partial"] = bool(failed or skipped)
    if not failed and not skipped:
        return
    parts: list[str] = []
    if failed:
        parts.append(f"{failed} request(s) failed")
    if skipped:
        parts.append(f"{skipped} rung(s) skipped")
    if invocation.machine_readable:
        write_document(invocation.streams.stdout, document)
    else:
        message_lines(invocation, probe_report_text(document))

    def hook() -> None:
        raise OperationFailed(
            f"The probe finished with {' and '.join(parts)}",
            hint=f"The run is saved; inspect it with provibench report {run_dir}",
            context={"run_dir": str(run_dir), "run_hex": run.run_hex},
        )

    invocation.on_success.append(hook)


def progress_line(record: ProbeResult) -> str:
    """One line per request; a TTL read names its offset, a stream names its TTFT."""
    what = f"ttl {record.attempt}s" if record.role == "ttl" else f"{record.role} {record.attempt}"
    message = (
        f"{record.spec_label} rung {record.rung} {what} "
        f"{record.status} {record.cached}/{record.prompt_total} {record.latency_ms:.0f}ms"
    )
    if record.ttft_ms is not None:
        message += f" ttft={record.ttft_ms:.0f}ms"
    if record.error:
        message += f" error={record.error[:_ERROR_TRUNCATE]}"
    return message
