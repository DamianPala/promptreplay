"""The scrub rules, the walker over entry strings, and what `--turns` selects."""

from __future__ import annotations

from typing import Any

import pytest

from promptreplay.bench.scrub import (
    PATH_RULE,
    SELECTION_ALL,
    SELECTION_CONVERSATION,
    SELECTION_ORDER,
    Replacement,
    Scrubber,
    ScrubError,
    collect_user_names,
    main_conversation_key,
    scrub_entry,
    select_entries,
)

_HOME_PATH = "/home/operator/notes.md"
_ANTHROPIC_KEY = "sk-ant-api03-FAKE0000000000000000000000"
_OPENROUTER_KEY = "sk-or-v1-FAKE0000000000000000000000"
_API_KEY = "sk-" + "A" * 24
_PROJECT_KEY = "sk-proj-FAKE000000000000000000000000"
_BEARER = "Bearer abcdefghijklmnop"
_AWS = "AKIAIOSFODNN7EXAMPLE"
_GITHUB = "ghp_FAKE0000000000000000000000"
_SLACK = "xoxb-0000-0000-FAKE"
_EMAIL = "operator@example.com"


def _scrub(text: str, **kwargs: object) -> tuple[str, Scrubber]:
    scrubber = Scrubber(**kwargs)  # type: ignore[arg-type]
    return scrubber.text(text), scrubber


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_ANTHROPIC_KEY, "[scrubbed:anthropic-key]"),
        (_OPENROUTER_KEY, "[scrubbed:api-key]"),
        (_API_KEY, "[scrubbed:api-key]"),
        (_PROJECT_KEY, "[scrubbed:api-key]"),
        (_BEARER, "Bearer [scrubbed:bearer]"),
        (_AWS, "[scrubbed:aws-key]"),
        (_GITHUB, "[scrubbed:github-token]"),
        (_SLACK, "[scrubbed:slack-token]"),
        (_EMAIL, "[scrubbed:email]"),
    ],
)
def test_each_rule_masks_its_secret(text: str, expected: str) -> None:
    scrubbed, scrubber = _scrub(f"value: {text}")
    assert scrubbed == f"value: {expected}"
    assert sum(1 for count in scrubber.counts.values() if count) == 1


def test_prose_that_merely_mentions_the_prefix_is_kept() -> None:
    prose = "The sk- prefix marks a key; sk-short is not one either."
    scrubbed, scrubber = _scrub(prose)
    assert scrubbed == prose
    assert scrubber.counts["api-key"] == 0


def test_home_paths_are_replaced_and_the_names_reported() -> None:
    scrubbed, scrubber = _scrub(f"read {_HOME_PATH} and /Users/other/there.md")
    assert scrubbed == "read /home/user/notes.md and /home/user/there.md"
    assert scrubber.counts[PATH_RULE] == 2
    assert scrubber.user_names == ["operator", "other"]


def test_a_home_path_that_ends_at_the_directory_is_replaced() -> None:
    """A system prompt's `cwd` and a recorded `HOME=` stop at the directory itself."""
    scrubbed, scrubber = _scrub("cwd /home/operator and HOME=/Users/operator")
    assert scrubbed == "cwd /home/user and HOME=/home/user"
    assert scrubber.counts[PATH_RULE] == 2
    assert scrubber.user_names == ["operator", "operator"]


def test_punctuation_after_a_home_path_is_kept() -> None:
    scrubbed, scrubber = _scrub("wrote /home/operator, then left")
    assert scrubbed == "wrote /home/user, then left"
    assert scrubber.user_names == ["operator"]


def test_a_nested_path_does_not_report_a_phantom_name() -> None:
    scrubbed, scrubber = _scrub("under /home/a/.config/home/b/x")
    assert scrubbed == "under /home/user/.config/home/b/x"
    assert scrubber.counts[PATH_RULE] == 1
    assert scrubber.user_names == ["a"]


def test_encoded_home_paths_are_rewritten() -> None:
    """Claude Code's state directory encodes `/home/<name>/x` as `-home-<name>-x`."""
    scrubbed, scrubber = _scrub(
        "memory /projects/-home-operator-recwork-acpc/memory/ and /home/operator/x"
    )
    assert scrubbed == "memory /projects/-home-user-recwork-acpc/memory/ and /home/user/x"
    assert scrubber.counts[PATH_RULE] == 2
    assert scrubber.user_names == ["operator"]


def test_encoded_home_paths_use_names_found_anywhere_in_the_trace() -> None:
    entries = [
        {"seq": 1, "body": {"text": "memory -home-operator-recwork-acpc/memory/"}},
        {"seq": 2, "body": {"text": "cwd /home/operator/recwork/acpc"}},
    ]
    scrubber = Scrubber(home_names=collect_user_names(entries))

    scrubbed = scrub_entry(entries[0], scrubber)

    assert scrubbed["body"] == {"text": "memory -home-user-recwork-acpc/memory/"}
    assert scrubber.counts[PATH_RULE] == 1


def test_user_names_are_replaced_as_whole_words() -> None:
    scrubbed, scrubber = _scrub("disk: -rw-rw-r-- 1 operator operator 42 f", users=("operator",))
    assert scrubbed == "disk: -rw-rw-r-- 1 user user 42 f"
    assert scrubber.counts["user"] == 2


def test_user_names_do_not_touch_longer_words() -> None:
    scrubbed, scrubber = _scrub("operators and operatorial", users=("operator",))
    assert scrubbed == "operators and operatorial"
    assert scrubber.counts["user"] == 0


def test_bearer_prose_is_not_masked() -> None:
    prose = "Use Bearer authentication when calling the API"
    scrubbed, scrubber = _scrub(prose)
    assert scrubbed == prose
    assert scrubber.counts["bearer"] == 0


def test_a_short_bearer_token_with_a_digit_is_masked() -> None:
    scrubbed, scrubber = _scrub("authorization: Bearer abc12345")
    assert scrubbed == "authorization: Bearer [scrubbed:bearer]"
    assert scrubber.counts["bearer"] == 1


def test_an_already_scrubbed_path_is_left_alone() -> None:
    scrubbed, scrubber = _scrub("read /home/user/notes.md")
    assert scrubbed == "read /home/user/notes.md"
    assert scrubber.counts[PATH_RULE] == 0
    assert scrubber.user_names == []


def test_allowed_emails_are_kept_and_the_rest_masked() -> None:
    scrubbed, scrubber = _scrub(
        f"from {_EMAIL} about other@example.com",
        allow_emails=("OPERATOR@example.com",),
    )
    assert scrubbed == f"from {_EMAIL} about [scrubbed:email]"
    assert scrubber.counts["email"] == 1


def test_the_walker_descends_into_nested_bodies_and_keys() -> None:
    body: dict[str, Any] = {
        "system": [{"type": "text", "text": f"See {_HOME_PATH}"}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": f"key {_API_KEY}"}],
            }
        ],
        _HOME_PATH: "a key that is itself a path",
    }
    scrubber = Scrubber()
    scrubbed = scrubber.data(body)
    assert scrubbed == {
        "system": [{"type": "text", "text": "See /home/user/notes.md"}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": "key [scrubbed:api-key]"}],
            }
        ],
        "/home/user/notes.md": "a key that is itself a path",
    }
    assert scrubber.counts[PATH_RULE] == 2


def test_literal_replacements_run_after_the_built_in_rules() -> None:
    text = f"project acme-corp holds {_EMAIL}"
    scrubbed, scrubber = _scrub(
        text, replacements=(Replacement("acme-corp", "example"), Replacement(_EMAIL, "gone"))
    )
    # The e-mail was masked by the built-in rule first, so the literal pair never matched.
    assert scrubbed == "project example holds [scrubbed:email]"
    assert scrubber.custom_counts == [1, 0]


def test_replacements_apply_in_order() -> None:
    scrubbed, scrubber = _scrub(
        "a b",
        replacements=(Replacement("a", "b"), Replacement("b", "c")),
    )
    assert scrubbed == "c c"
    assert scrubber.custom_counts == [1, 2]


def test_metadata_is_removed_from_the_body_only() -> None:
    entry = {
        "seq": 1,
        "body": {
            "metadata": {"user_id": "operator"},
            "system": [{"type": "text", "text": _HOME_PATH, "metadata": {"user_id": "operator"}}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": _HOME_PATH, "metadata": {"id": "kept"}}
                    ],
                }
            ],
        },
    }
    scrubber = Scrubber()
    scrubbed = scrub_entry(entry, scrubber)
    body = scrubbed["body"]
    assert isinstance(body, dict)
    assert "metadata" not in body
    assert scrubber.counts["metadata"] == 1
    # Nested `metadata` blocks stay: only the request's own metadata is dropped, and the
    # strings inside every block are scrubbed like any other.
    assert body["system"] == [
        {"type": "text", "text": "/home/user/notes.md", "metadata": {"user_id": "operator"}}
    ]
    assert body["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "content": "/home/user/notes.md",
                    "metadata": {"id": "kept"},
                }
            ],
        }
    ]


def test_scrubbing_is_deterministic() -> None:
    entry = {
        "headers": {"authorization": _BEARER},
        "body": {"messages": [{"role": "user", "content": f"{_HOME_PATH} {_API_KEY}"}]},
    }
    first = scrub_entry(entry, Scrubber())
    second = scrub_entry(entry, Scrubber())
    assert first == second


def _entry(
    seq: int, conversation: str, size: int, *, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "seq": seq,
        "conversation": conversation,
        "body": body
        if body is not None
        else {"messages": [{"role": "user", "content": "x" * size}]},
    }


def test_select_entries_keeps_everything_without_turns() -> None:
    entries = [_entry(1, "side", 10), _entry(2, "main", 1000)]
    kept, selection = select_entries(entries, turns=None)
    assert kept == entries
    assert selection == SELECTION_ALL


def test_select_entries_keeps_the_first_turns_of_the_main_conversation() -> None:
    entries = [
        _entry(1, "side", 10),
        _entry(2, "main", 1000),
        _entry(3, "main", 1000),
        _entry(4, "main", 1000),
        _entry(5, "side", 10),
    ]
    kept, selection = select_entries(entries, turns=2)
    assert [entry["seq"] for entry in kept] == [2, 3]
    assert selection == SELECTION_CONVERSATION


def test_select_entries_falls_back_to_file_order_without_conversations() -> None:
    entries = [{"seq": seq, "body": {"text": "x" * 10}} for seq in (1, 2, 3)]

    kept, selection = select_entries(entries, turns=2)

    assert [entry["seq"] for entry in kept] == [1, 2]
    assert selection == SELECTION_ORDER


def test_select_entries_on_an_empty_trace() -> None:
    assert select_entries([], turns=3) == ([], SELECTION_ORDER)
    assert main_conversation_key([]) is None


def test_main_conversation_is_the_one_with_the_most_request_bytes() -> None:
    entries = [_entry(1, "side", 10), _entry(2, "main", 1000), _entry(3, "main", 1000)]
    assert main_conversation_key(entries) == "main"


def test_report_counts_every_rule_and_the_user_names() -> None:
    scrubber = Scrubber(replacements=(Replacement("acme-corp", "example"),), users=("operator",))
    scrubber.data({"body": {"text": f"{_HOME_PATH} operator {_EMAIL} acme-corp"}})
    report = scrubber.report(entries_in=4, entries_out=3, bytes_in=100, bytes_out=60)
    counts = {row.rule: row.count for row in report.rules}
    assert counts[PATH_RULE] == 1
    assert counts["email"] == 1
    assert counts["user"] == 1  # the name inside the path was already replaced
    assert counts["replace:acme-corp"] == 1
    assert counts["aws-key"] == 0
    assert list(counts) == [
        "path",
        "metadata",
        "anthropic-key",
        "openrouter-key",
        "api-key",
        "bearer",
        "aws-key",
        "github-token",
        "slack-token",
        "email",
        "user",
        "replace:acme-corp",
    ]
    assert report.user_names == ["operator"]
    assert report.entries_dropped == 1
    assert report.changed is True


def test_colliding_keys_are_an_error_naming_the_entry() -> None:
    entry = {"seq": 7, "body": {"/home/a/x": 1, "/home/b/x": 2}}

    with pytest.raises(ScrubError) as raised:
        scrub_entry(entry, Scrubber())

    assert "7" in str(raised.value)
    assert "/home/user/x" in str(raised.value)
