"""JSON documents and the O4 JSON Schema subset used in D8 `output` declarations."""

from typing import TypedDict, cast

type Document = dict[str, object]
"""One JSON object as the tool emits it: a success document, a record, or an F3 envelope."""


def as_document(value: object) -> Document | None:
    """`value` as a document when it is a JSON object, else `None`; the typed JSON boundary."""
    if isinstance(value, dict):
        return cast(Document, value)
    return None


def as_list(value: object) -> list[object] | None:
    """`value` as a list when it is a JSON array, else `None`; the typed JSON boundary."""
    if isinstance(value, list):
        return cast(list[object], value)
    return None


class JsonSchema(TypedDict, total=False):
    """The O4 subset of JSON Schema: only `type`, `enum`, `properties`, `required`, `items`."""

    type: str | list[str]
    enum: list[object]
    properties: dict[str, "JsonSchema"]
    required: list[str]
    items: "JsonSchema"


def string(*, enum: list[str] | None = None) -> JsonSchema:
    """A string schema, optionally restricted to a closed set."""
    if enum is None:
        return {"type": "string"}
    return {"type": "string", "enum": list(enum)}


def nullable_string() -> JsonSchema:
    """A string schema that also admits `null`."""
    return {"type": ["string", "null"]}


def nullable_integer() -> JsonSchema:
    """An integer schema that also admits `null`."""
    return {"type": ["integer", "null"]}


def integer() -> JsonSchema:
    """An integer schema."""
    return {"type": "integer"}


def number() -> JsonSchema:
    """A number schema (integer or floating point)."""
    return {"type": "number"}


def nullable_number() -> JsonSchema:
    """A number schema that also admits `null`."""
    return {"type": ["number", "null"]}


def boolean() -> JsonSchema:
    """A boolean schema."""
    return {"type": "boolean"}


def nullable_boolean() -> JsonSchema:
    """A boolean schema that also admits `null`."""
    return {"type": ["boolean", "null"]}


def array(items: JsonSchema) -> JsonSchema:
    """An array schema with one item schema."""
    return {"type": "array", "items": items}


def obj(properties: dict[str, JsonSchema], *, required: list[str]) -> JsonSchema:
    """An object schema listing every field the command may return and the always-present ones."""
    return {"type": "object", "required": required, "properties": properties}


def nullable_object(properties: dict[str, JsonSchema], *, required: list[str]) -> JsonSchema:
    """An object schema admitting `null`, otherwise as `obj`."""
    return {"type": ["object", "null"], "required": required, "properties": properties}


def collection(item: JsonSchema) -> JsonSchema:
    """The O6 and C1 page shape: `items`, `has_more`, and an always-present `next_cursor`."""
    return obj(
        {"items": array(item), "has_more": boolean(), "next_cursor": nullable_string()},
        required=["items", "has_more", "next_cursor"],
    )


def page_document(items: list[Document], next_cursor: str | None) -> Document:
    """A document of the `collection` shape: `next_cursor` is a string exactly when `has_more`."""
    return {"items": items, "has_more": next_cursor is not None, "next_cursor": next_cursor}


def sanitize_document(document: Document) -> Document:
    """`document` with every lone surrogate in a string replaced by U+FFFD (O5a).

    A path or other OS-supplied string decoded with `surrogateescape` (an invalid config
    or state path, for instance) can carry an unpaired surrogate. Left alone, that
    surrogate either fails to encode as UTF-8 or, if the stream itself uses
    `surrogateescape`, comes back out as the original invalid byte. Replacing it here,
    at the single point where a document is about to be serialized, keeps every writer
    honest without touching the commands that produced the value.
    """
    return cast(Document, _sanitize_value(document))


def _sanitize_value(value: object) -> object:
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, dict):
        items = cast("dict[object, object]", value)
        return {_sanitize_value(key): _sanitize_value(item) for key, item in items.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in cast("list[object]", value)]
    return value


def _sanitize_string(value: str) -> str:
    if value.isascii():
        return value
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return "".join("�" if 0xD800 <= ord(char) <= 0xDFFF else char for char in value)
    return value
