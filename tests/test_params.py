"""Input contracts for standard durations and backwards-compatible timeouts."""

import pytest
from click import BadParameter

from promptreplay.core.documents import as_document, as_list
from promptreplay.core.params import TIMEOUT
from tests.conftest import Cli


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("30s", 30.0), ("5m", 300.0), ("2h", 7200.0), ("30d", 2_592_000.0), ("0.2", 0.2)],
)
def test_timeout_accepts_standard_durations_and_decimal_seconds(value: str, seconds: float) -> None:
    assert TIMEOUT.convert(value, None, None) == seconds


@pytest.mark.parametrize("value", ["0s", "0", "fast", "inf"])
def test_timeout_rejects_invalid_values(value: str) -> None:
    with pytest.raises(BadParameter):
        TIMEOUT.convert(value, None, None)


def test_probe_accepts_legacy_decimal_timeout_and_schema_uses_duration_default(cli: Cli) -> None:
    outcome = cli.run("probe", "missing", "deepseek:model", "--timeout", "30")
    assert outcome.code == 1
    assert outcome.error["kind"] == "not_found"

    detail = cli.run("schema", "probe").document
    flags = {
        str(flag["name"]): flag for flag in map(as_document, as_list(detail["flags"]) or []) if flag
    }
    assert flags["timeout"]["default"] == "300s"
