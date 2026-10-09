"""chat.js send path: names the catch block reads must not be try-scoped.

`_isAgent` and `roundHolder` were declared with const/let inside the big
`try { ... }` of the send function but read by its `catch`: a response
timeout showed "_isAgent is not defined" instead of the timeout message,
and the terminal-error fallback could throw on `roundHolder`.
"""
import re
from pathlib import Path

CHAT_JS = Path(__file__).resolve().parents[1] / "static" / "js" / "chat.js"


def _send_try_head() -> str:
    """Source from the send state declarations up to the big try."""
    src = CHAT_JS.read_text(encoding="utf-8")
    start = src.index("let streamingTTS = false;")
    end = src.index("try {", start)
    return src[start:end]


def test_catch_read_names_are_declared_before_the_try():
    head = _send_try_head()
    assert re.search(r"\blet _isAgent\b", head)
    assert re.search(r"\blet roundHolder\b", head)


def _send_function() -> str:
    """Source of the send function: from its state declarations up to the
    next function declared at the same (module-closure) level."""
    src = CHAT_JS.read_text(encoding="utf-8")
    start = src.index("let streamingTTS = false;")
    end = re.compile(r"^  (?:async )?function ", re.M).search(src, start)
    return src[start:end.start() if end else len(src)]


def test_no_try_scoped_redeclarations_remain():
    # Other functions (e.g. the stream-resume reader) keep their own
    # function-level roundHolder; only the send function's try is at issue.
    body = _send_function()
    assert not re.search(r"\b(const|let|var)\s+_isAgent\s*=\s*\(", body)
    assert "let roundHolder = holder" not in body
