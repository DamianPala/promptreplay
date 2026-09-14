"""What makes a trace shareable: the masking rules and the walker over its strings.

A trace records request bytes verbatim, so it carries home paths, the operator's
instruction files (and the harness's encoded copy of the project path),
`metadata.user_id` and possibly API keys. Everything here is pure: the command does the
file I/O, this module decides what a scrubbed entry looks like.

Rules run in a fixed order — user paths, the secret table, `--user` names, then the
caller's literal replacements — so the same input and options always produce the same
output.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

PATH_RULE = "path"
"""The report's name for home paths; `/home/<name>` and `-home-<name>-` both count here."""

METADATA_RULE = "metadata"
"""The report's name for the `body.metadata` fields dropped from the entries."""

USER_RULE = "user"
"""The report's name for the `--user NAME` whole-word replacements."""

EMAIL_RULE = "email"
"""The one rule whose matches `--allow-email` can keep."""

BEARER_RULE = "bearer"
"""The one rule that has to tell a credential apart from prose."""

HOME_RE = re.compile(r"(?<!\w)/(?:home|Users)/([A-Za-z0-9._-]+)")
"""`/home/<name>` and `/Users/<name>`; group 1 is the name, reported but not written.

The slash after the name is not required: a system prompt's `cwd` and a recorded `HOME=`
end at the directory itself. The name is a single path segment of name characters, so the
punctuation that follows a path in prose stays where it was, and the match may not start
inside a word, so `/home/a/.config/home/b` reports one name, not two.
"""

HOME_PLACEHOLDER = "/home/user"
"""What every matched home path becomes, whatever the user was called."""

ENCODED_HOME = "-home-{name}-"
"""Claude Code encodes a project path into its state directory: `/home/haz/x` as `-home-haz-x`."""

ENCODED_PLACEHOLDER = "-home-user-"
"""What the encoded form becomes."""

USER_WORD = "user"
"""What `--user NAME` replaces the name with."""


@dataclass(frozen=True, slots=True)
class Rule:
    """One masking rule: what the report calls it, its F3 kind, and what it matches."""

    label: str
    kind: str
    pattern: re.Pattern[str]


SECRET_RULES: tuple[Rule, ...] = (
    Rule("anthropic-key", "anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]+")),
    Rule("openrouter-key", "api-key", re.compile(r"\bsk-or-[A-Za-z0-9_-]+")),
    # 20+ key characters, so prose that merely mentions `sk-` stays readable. Hyphens and
    # underscores count: OpenAI's current keys are `sk-proj-…` and `sk-svcacct-…`.
    Rule("api-key", "api-key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    # Only the token is masked; the `Bearer` scheme stays, as `Authorization: Bearer x`.
    Rule(BEARER_RULE, "bearer", re.compile(r"(?<=\bBearer )[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)),
    Rule("aws-key", "aws-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    Rule("github-token", "github-token", re.compile(r"\b(?:ghp_|gho_|github_pat_)[A-Za-z0-9_]+")),
    Rule("slack-token", "slack-token", re.compile(r"\bxox[abp]-[A-Za-z0-9-]+")),
    Rule(EMAIL_RULE, "email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
)
"""The secret patterns, in the order they are applied. Extend by adding a row here."""

SELECTION_ALL = "all"
"""`--turns` was not given: every entry is written."""

SELECTION_CONVERSATION = "conversation"
"""`--turns` kept the first N entries of the main conversation."""

SELECTION_ORDER = "order"
"""`--turns` kept the first N entries in file order: no entry carried a conversation."""


def scrubbed(kind: str) -> str:
    """The placeholder that replaces one secret."""
    return f"[scrubbed:{kind}]"


@dataclass(frozen=True, slots=True)
class Replacement:
    """One `--replace OLD=NEW` pair, applied literally after the built-in rules."""

    old: str
    new: str


def replacement_label(replacement: Replacement) -> str:
    """The report's rule name for one `--replace` pair."""
    return f"replace:{replacement.old}"


class ScrubError(ValueError):
    """A scrub cannot represent what it was given any more."""


class KeyCollision(ScrubError):
    """Two dict keys scrub to the same string, so one of them would be lost."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


class Scrubber:
    """One pass over a trace: what each rule replaced, and the user names it saw."""

    def __init__(
        self,
        *,
        allow_emails: Iterable[str] = (),
        replacements: Sequence[Replacement] = (),
        users: Iterable[str] = (),
        home_names: Iterable[str] = (),
    ) -> None:
        self._allowed_emails = {address.lower() for address in allow_emails}
        self._replacements = tuple(replacements)
        self._user_patterns = [re.compile(rf"\b{re.escape(name)}\b") for name in users if name]
        # Names the path rule found anywhere in the trace, so the encoded form is rewritten
        # whatever the order of the strings it appears in; see `collect_user_names`.
        self.user_names: list[str] = list(dict.fromkeys(home_names))
        self.counts: dict[str, int] = {
            PATH_RULE: 0,
            METADATA_RULE: 0,
            **{rule.label: 0 for rule in SECRET_RULES},
            USER_RULE: 0,
        }
        self.custom_counts: list[int] = [0] * len(self._replacements)

    def report(
        self, *, entries_in: int, entries_out: int, bytes_in: int, bytes_out: int
    ) -> ScrubReport:
        """The report of this pass: every rule, in the order the walker applies them."""
        rules = [RuleCount(label, count) for label, count in self.counts.items()]
        rules += [
            RuleCount(replacement_label(replacement), count)
            for replacement, count in zip(self._replacements, self.custom_counts, strict=True)
        ]
        return ScrubReport(
            rules=rules,
            user_names=sorted(set(self.user_names)),
            entries_in=entries_in,
            entries_out=entries_out,
            bytes_in=bytes_in,
            bytes_out=bytes_out,
        )

    def text(self, value: str) -> str:
        """`value` with every home path, secret, user name and literal replacement applied."""
        value = self._mask_user_paths(value)
        for rule in SECRET_RULES:
            value = rule.pattern.sub(partial(self._mask, rule), value)
        for pattern in self._user_patterns:
            value, found = pattern.subn(USER_WORD, value)
            self.counts[USER_RULE] += found
        for index, replacement in enumerate(self._replacements):
            found = value.count(replacement.old)
            if found:
                self.custom_counts[index] += found
                value = value.replace(replacement.old, replacement.new)
        return value

    def data(self, value: object) -> object:
        """`value` with every string scrubbed, recursively; keys included.

        A key that scrubs to the same string as another key is a `KeyCollision`: keeping
        either one would silently drop the other's value.
        """
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.data(item) for item in cast("list[object]", value)]
        if isinstance(value, dict):
            items = cast("dict[object, object]", value)
            keys = [self.text(str(key)) for key in items]
            duplicate = _first_duplicate(keys)
            if duplicate is not None:
                raise KeyCollision(duplicate)
            return {key: self.data(item) for key, item in zip(keys, items.values(), strict=True)}
        return value

    def _mask(self, rule: Rule, match: re.Match[str]) -> str:
        matched = match.group(0)
        if rule.label == EMAIL_RULE and matched.lower() in self._allowed_emails:
            return matched
        if rule.label == BEARER_RULE and not _is_token(matched):
            return matched
        self.counts[rule.label] += 1
        return scrubbed(rule.kind)

    def _mask_user_paths(self, value: str) -> str:
        """`/home/<operator>` as `/home/user`, collecting the names for the report."""
        parts: list[str] = []
        cursor = 0
        for match in HOME_RE.finditer(value):
            name = match.group(1)
            if name == "user":
                # Already scrubbed (or a real account called `user`): nothing to hide.
                continue
            parts.append(value[cursor : match.start()])
            parts.append(HOME_PLACEHOLDER)
            cursor = match.end()
            self.counts[PATH_RULE] += 1
            self.user_names.append(name)
        if parts:
            parts.append(value[cursor:])
            value = "".join(parts)
        return self._mask_encoded_paths(value)

    def _mask_encoded_paths(self, value: str) -> str:
        """`-home-<name>-` (Claude Code's encoded cwd) as `-home-user-`."""
        for name in dict.fromkeys(self.user_names):
            encoded = ENCODED_HOME.format(name=name)
            found = value.count(encoded)
            if found:
                self.counts[PATH_RULE] += found
                value = value.replace(encoded, ENCODED_PLACEHOLDER)
        return value


def scrub_entry(entry: dict[str, Any], scrubber: Scrubber) -> dict[str, Any]:
    """One entry with every string scrubbed and `body.metadata` removed.

    `metadata` carries the operator's `user_id`; it is dropped rather than masked, since
    a replayed request does not need it.
    """
    try:
        document = cast("dict[str, Any]", scrubber.data(entry))
    except KeyCollision as exc:
        raise ScrubError(
            f"Entry {entry.get('seq', '?')}: scrubbing gives two keys the name {exc.key!r}"
        ) from exc
    body = document.get("body")
    if isinstance(body, dict) and "metadata" in cast("dict[str, Any]", body):
        cast("dict[str, Any]", body).pop("metadata")
        scrubber.counts[METADATA_RULE] += 1
    return document


def collect_user_names(documents: Sequence[dict[str, Any]]) -> list[str]:
    """Every `/home/<name>` (or `/Users/<name>`) name in the documents, first-seen order.

    Run over everything about to be scrubbed before the scrub itself, so `-home-<name>-`
    is rewritten wherever it appears, whatever the order of the strings holding it.
    """
    names: dict[str, None] = {}
    for document in documents:
        for text in _strings(document):
            for match in HOME_RE.finditer(text):
                name = match.group(1)
                if name != "user":
                    names.setdefault(name, None)
    return list(names)


def _strings(value: object) -> Iterator[str]:
    """Every string in a JSON value, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in cast("list[object]", value):
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in cast("dict[object, object]", value).items():
            yield str(key)
            yield from _strings(item)


def _first_duplicate(keys: list[str]) -> str | None:
    """The first key that repeats, or `None` when every key is distinct."""
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            return key
        seen.add(key)
    return None


def _is_token(value: str) -> bool:
    """Whether a `Bearer` value is a credential rather than prose.

    "Use Bearer authentication" is prose; a token carries a digit or a symbol, or is long
    enough that it cannot be an English word.
    """
    return len(value) >= 16 or not value.isalpha()


def select_entries(
    entries: list[dict[str, Any]], *, turns: int | None
) -> tuple[list[dict[str, Any]], str]:
    """The entries to write, and which rule chose them.

    Without `turns` every entry is kept (`all`). With it, the first `turns` entries of the
    main conversation survive and every other conversation is dropped (`conversation`). A
    trace whose entries carry no conversation at all keeps its first `turns` entries in
    file order (`order`) rather than coming out empty.
    """
    if turns is None:
        return list(entries), SELECTION_ALL
    if not any(_conversation(entry) for entry in entries):
        return entries[:turns], SELECTION_ORDER
    main = main_conversation_key(entries)
    kept = [entry for entry in entries if _conversation(entry) == main][:turns]
    return kept, SELECTION_CONVERSATION


def main_conversation_key(entries: list[dict[str, Any]]) -> str | None:
    """The conversation carrying the most request bytes: the agent's main loop.

    The same choice `bench.trace.main_conversation` makes for a parsed trace, on the raw
    entry documents a scrub reads and writes.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(_conversation(entry), []).append(entry)
    if not groups:
        return None
    return max(groups, key=lambda key: sum(_body_bytes(entry) for entry in groups[key]))


def _conversation(entry: dict[str, Any]) -> str:
    """An entry's conversation key: a missing or null one is the empty group."""
    value = entry.get("conversation")
    return "" if value is None else str(value)


def _body_bytes(entry: dict[str, Any]) -> int:
    """A request body's serialized size, as `TraceEntry.body_bytes` measures it."""
    return len(json.dumps(entry.get("body"), ensure_ascii=False))


@dataclass(frozen=True, slots=True)
class RuleCount:
    """One row of the report: a rule and how many times it fired."""

    rule: str
    count: int


@dataclass(frozen=True, slots=True)
class ScrubReport:
    """What one scrub changed: per rule, plus the entries and bytes it touched."""

    rules: list[RuleCount]
    user_names: list[str]
    entries_in: int
    entries_out: int
    bytes_in: int
    bytes_out: int

    @property
    def entries_dropped(self) -> int:
        """Entries `--turns` left out."""
        return self.entries_in - self.entries_out

    @property
    def changed(self) -> bool:
        """Whether anything was removed: entries dropped, or a rule that fired.

        Not a byte comparison: re-serialising an input written with spaces changes
        `bytes_out` on its own, and this is the field someone checks before sharing.
        """
        return self.entries_dropped > 0 or any(row.count for row in self.rules)
