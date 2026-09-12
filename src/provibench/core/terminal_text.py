"""Make control characters in untrusted terminal text visible (O3d)."""


def escape_terminal_text(text: str) -> str:
    """Replace every C0, C1, and DEL character with its visible hexadecimal spelling."""
    return "".join(_visible(character) for character in text)


def _visible(character: str) -> str:
    codepoint = ord(character)
    if codepoint <= 0x1F or 0x7F <= codepoint <= 0x9F:
        return f"\\x{codepoint:02x}"
    return character
