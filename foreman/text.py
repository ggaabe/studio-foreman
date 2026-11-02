"""Text hygiene for untrusted strings: LLM replies, inbox notes, file names, brain notes.

Two jobs:

  * Compare what a string SAYS, not how it is encoded. `fold` applies NFKC (fullwidth letters,
    ligatures and other compatibility forms become their plain spelling) and removes characters
    that render as nothing: Unicode format characters (category Cf: zero-width space and joiners,
    the byte-order mark, soft hyphen, word joiner, bidi controls, tag characters), Hangul and
    braille fillers, the combining grapheme joiner and variation selectors. Content hashes,
    consumer matching and the banned-class and URL filters all work on folded text, so a copy
    that differs only in invisible characters is the same copy.
  * Print untrusted text without letting it drive a terminal. `printable` shows control
    characters, line breaks and invisible or bidi formatting characters as escapes, so an OSC 52
    clipboard write, a carriage return or a cursor move in a brief cannot overwrite or forge what
    the owner reads.
"""
from __future__ import annotations

import re
import unicodedata

# Characters that show as nothing (or as a blank that is not whitespace) without being Cf.
_FILLERS = frozenset("ᅟᅠㅤﾠ⠀͏឴឵")
# C0 and C1 controls other than tab and newline.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _invisible(ch: str) -> bool:
    if ch in _FILLERS:
        return True
    o = ord(ch)
    if 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF:  # variation selectors
        return True
    return unicodedata.category(ch) == "Cf"


def strip_invisible(text: str, repl: str = "") -> str:
    """`text` with every invisible character removed (or replaced by `repl`)."""
    if text.isascii():
        return text
    return "".join(repl if _invisible(ch) else ch for ch in text)


def fold(text: str) -> str:
    """NFKC without invisible characters: the spelling filters and hashes compare."""
    if text.isascii():
        return text
    return strip_invisible(unicodedata.normalize("NFKC", strip_invisible(text)))


def variants(text: str) -> tuple:
    """Spellings a filter should try: invisible characters removed, invisible characters read as
    a space ('mass<word joiner>SEO' is 'mass SEO'), and the folded text without combining marks."""
    a = fold(text)
    if text.isascii():
        return (a,)
    b = unicodedata.normalize("NFKC", strip_invisible(text, " "))
    c = "".join(ch for ch in unicodedata.normalize("NFKD", a) if not unicodedata.combining(ch))
    return tuple(dict.fromkeys((a, b, c)))


# Letters of other scripts that render like a Latin letter (Cyrillic, Greek and a few IPA forms):
# 'guarantееd' with two Cyrillic 'е' reads as 'guaranteed' on screen and in the voice. `skeleton`
# maps them to the Latin letter so a filter compares what a viewer sees.
_CONFUSABLES = {
    "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "с": "c", "т": "t",
    "у": "y", "х": "x", "ѕ": "s", "і": "i", "ї": "i", "ј": "j", "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w",
    "ɡ": "g", "ɩ": "i", "ʏ": "y", "ᴀ": "a", "ο": "o", "α": "a", "ν": "v", "ρ": "p", "ι": "i", "κ": "k", "τ": "t",
    "υ": "u", "χ": "x", "ε": "e", "ϲ": "c", "А": "A", "В": "B", "Е": "E", "Ё": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ј": "J", "Ԛ": "Q", "Ԝ": "W",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Υ": "Y", "Χ": "X", "Ϲ": "C",
}
_CONFUSABLE_TABLE = str.maketrans(_CONFUSABLES)


def skeleton(text: str) -> str:
    """`fold(text)` with look-alike letters of other scripts read as the Latin letter they show."""
    t = fold(text)
    return t if t.isascii() else t.translate(_CONFUSABLE_TABLE)


def _script(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    if ch.isascii():
        return "latin"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "other"
    for s in ("LATIN", "CYRILLIC", "GREEK"):
        if name.startswith(s):
            return s.lower()
    return "other"


def mixed_script_words(text: str) -> list:
    """Words that mix Latin letters with Cyrillic or Greek ones (a homoglyph spelling such as
    'ОtherCo' with a Cyrillic 'О'): no real word of any language does that."""
    out = []
    for word in re.findall(r"[^\W\d_]+", fold(text)):
        scripts = {_script(ch) for ch in word} - {None}
        if "latin" in scripts and scripts & {"cyrillic", "greek"}:
            out.append(word)
    return out


BIDI_CONTROLS = frozenset("؜‎‏‪‫‬‭‮⁦⁧⁨⁩")


def has_bidi(text: str) -> bool:
    """Characters that reorder how a line is displayed (what a viewer reads is not the order a
    filter reads it in)."""
    return any(ch in BIDI_CONTROLS for ch in str(text))


def has_control(text: str) -> bool:
    """A C0 or C1 control character other than tab and newline."""
    return bool(CONTROL_RE.search(text))


def printable(text, limit: int | None = None) -> str:
    """One line that is safe to print on a terminal or to put in a Markdown cell: control
    characters, line breaks and invisible or bidi formatting characters are shown as escapes."""
    s = str(text)
    if limit is not None:
        s = s[:limit]
    if s.isascii() and s.isprintable():
        return s
    out = []
    for ch in s:
        if unicodedata.category(ch) == "Zs":
            out.append(" ")
        elif ch.isprintable() and not _invisible(ch):
            out.append(ch)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x100 and not ch.isprintable():
            out.append(f"\\x{ord(ch):02x}")
        elif ord(ch) < 0x10000:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(f"\\U{ord(ch):08x}")
    return "".join(out)
