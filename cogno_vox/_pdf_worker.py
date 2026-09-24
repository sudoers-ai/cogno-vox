"""The process that reads an UNTRUSTED PDF — run ONLY by ``cogno_vox.pdf_text``, never imported
for its effects.

The parent (``PdfTextExtractor``) runs this file's SOURCE with ``python -E -c`` in an empty
temporary directory, hands it the bytes on stdin and reads one JSON line from stdout. It is a
module of its own so it can be linted, type-checked and unit-tested like any other; it imports
nothing from ``cogno_vox`` so that running it needs nothing on ``sys.path`` but ``pypdfium2``.

**The sandbox, applied AFTER the parser is loaded and BEFORE a byte of the file is read**
(:func:`sandbox`):

* every descriptor above stdio CLOSED (nothing inherited — an inherited socket would be network
  without opening anything) and ``RLIMIT_NOFILE`` set to 3 — the process can open NO new file
  descriptor, which means no socket (network), no file and no pipe. This is the
  no-network rule enforced by the KERNEL, for the native parser too, not a Python patch around
  it (the Python ``socket`` is ALSO replaced, for a readable error). Because nothing can be
  opened afterwards, every lazily-loaded path is exercised first on a document of our own
  (:func:`warm_up`);
* ``RLIMIT_AS`` — the address space a malicious file can make the parser balloon to;
* ``RLIMIT_CPU`` — a spinning parser is killed by the kernel even if the parent died;
* ``RLIMIT_FSIZE`` 0 — nothing may be written to a file;
* ``RLIMIT_NPROC`` 0 — nothing may be forked.

**The text layer only**: the document is opened from memory, forms are never initialised (so no
JavaScript runs), attachments are never read, links are never followed, nothing is rendered.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Kept as plain strings, not imported: this file runs with nothing of cogno_vox on the path.
NO_TEXT = "no_text"
OVER_LIMIT = "over_limit"
ENCRYPTED = "encrypted"
INVALID = "invalid"
TIMEOUT = "timeout"

#: PDFium's error code for a document that needs a password (``FPDF_ERR_PASSWORD``).
_PDFIUM_ERR_PASSWORD = 4
#: Bookmarks read at most — an outline is navigation, a million of them is an attack.
MAX_OUTLINE = 2000


def _refuse(*_a: Any, **_k: Any) -> Any:
    raise PermissionError("network is not available to the PDF worker")


def sandbox(*, memory_bytes: int, cpu_seconds: int) -> None:  # pragma: no cover — worker process only
    """Close every door this process does not need — see the module docstring."""
    import resource
    import socket

    socket.socket = _refuse  # type: ignore[assignment,misc]
    socket.create_connection = _refuse
    socket.getaddrinfo = _refuse
    # Two different doors. The limit of 3 (last line) refuses every NEW descriptor: 0, 1 and 2
    # are taken, and a new one would have to be numbered below the limit. It does nothing about
    # a descriptor the process INHERITED — an open socket handed down by whatever spawned it is
    # network without opening anything. So everything above stdio is closed first. The parent
    # here spawns with `close_fds`, so nothing should arrive; this holds even if something does.
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    os.closerange(3, soft if 0 < soft < 1 << 20 else 4096)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, OSError):
        pass                                        # not every platform lets it drop to 0
    # LAST: after this nothing new can be opened, including by the calls above.
    resource.setrlimit(resource.RLIMIT_NOFILE, (3, 3))


def read_text(pdfium: Any, data: bytes, *, max_pages: int, max_chars: int) -> dict:
    """``{"pages": [...], "outline": [...]}`` or ``{"error": reason}`` — the whole contract of
    one extraction, in-process (the unit tests call it directly)."""
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        return {"error": ENCRYPTED if getattr(exc, "err_code", None) == _PDFIUM_ERR_PASSWORD
                else INVALID}
    try:
        count = len(pdf)
        if count > max_pages:
            return {"error": OVER_LIMIT}        # counted from the page tree; no page was read
        pages: "list[dict]" = []
        total = 0
        for index in range(count):
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_range()
                finally:
                    textpage.close()
            finally:
                page.close()
            total += len(text)
            if total > max_chars:
                return {"error": OVER_LIMIT}
            pages.append({"number": index + 1, "text": text})
        if not any(p["text"].strip() for p in pages):
            return {"error": NO_TEXT}
        outline: "list[dict]" = []
        for item in pdf.get_toc():
            if len(outline) >= MAX_OUTLINE:
                break
            dest = item.get_dest()
            target = dest.get_index() if dest is not None else None
            title = (item.get_title() or "").strip()
            if title and target is not None:
                outline.append({"level": int(item.level) + 1, "title": title,
                                "page": int(target) + 1})
        return {"pages": pages, "outline": outline}
    finally:
        pdf.close()


def tiny_pdf(texts: "list[str]") -> bytes:
    """A minimal valid PDF, one Helvetica line per page (ASCII) — the warm-up document, and what
    the tests build their fixtures with."""
    objs: "list[bytes]" = []

    def add(body: bytes) -> int:
        objs.append(body)
        return len(objs)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    tree = add(b"")
    kids = []
    for text in texts:
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = (b"BT /F1 12 Tf 72 720 Td (" + esc.encode("latin-1") + b") Tj ET") if text else b""
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        kids.append(add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                        b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                        % (tree, font, content)))
    objs[tree - 1] = (b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % k for k in kids)
                      + b"] /Count %d >>" % len(kids))
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % tree)
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += (b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, catalog, xref))
    return out


def warm_up(pdfium: Any) -> None:
    """Run every path of :func:`read_text` once, on a document of our own, BEFORE the sandbox.

    The parser and the interpreter both load code LAZILY — measured: the first
    ``get_text_range`` imports the ``utf_16_le`` codec, which opens a file — and once the
    descriptor limit is in place nothing can be opened any more. A path first taken inside the
    sandbox would fail as ``invalid`` on a perfectly good file."""
    read_text(pdfium, tiny_pdf(["warm up", ""]), max_pages=10, max_chars=1000)
    read_text(pdfium, b"%PDF-1.4 not really", max_pages=10, max_chars=1000)
    json.dumps({"x": "é"}, ensure_ascii=False)


def selfcheck() -> dict:  # pragma: no cover — worker process only
    """What the sandbox refuses, measured from inside it — the test that the doors are shut."""
    out: dict = {}
    import _socket
    import socket
    try:
        socket.socket()
        out["socket"] = "allowed"
    except Exception as exc:                      # noqa: BLE001
        out["socket"] = type(exc).__name__
    try:                                          # the C level, under the Python patch: the KERNEL
        _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM).close()
        out["raw_socket"] = "allowed"
    except OSError as exc:
        out["raw_socket"] = f"OSError:{exc.errno}"
    try:
        fd = os.open("/dev/null", os.O_RDONLY)
        os.close(fd)
        out["open"] = "allowed"
    except OSError as exc:
        out["open"] = type(exc).__name__
    inherited = []
    for fd in range(3, 256):
        try:
            os.fstat(fd)
            inherited.append(fd)
        except OSError:
            pass
    out["inherited"] = inherited
    try:
        blob = bytearray(4 * 1024 * 1024 * 1024)      # far past any memory ceiling we set
        out["memory"] = "allowed" if blob else "allowed"
    except MemoryError:
        out["memory"] = "MemoryError"
    return out


def main(argv: "list[str]") -> int:  # pragma: no cover — worker process only
    mode, max_pages, memory_bytes, cpu_seconds, max_chars = argv[:5]
    import pypdfium2 as pdfium                       # BEFORE the sandbox: loading it opens files
    warm_up(pdfium)
    sandbox(memory_bytes=int(memory_bytes), cpu_seconds=int(cpu_seconds))
    if mode == "selfcheck":
        result = selfcheck()
    else:
        data = sys.stdin.buffer.read()
        try:
            result = read_text(pdfium, data, max_pages=int(max_pages), max_chars=int(max_chars))
        except MemoryError:
            result = {"error": OVER_LIMIT}
        except Exception:                            # noqa: BLE001 — a parser crash is `invalid`
            result = {"error": INVALID}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover — the worker entry point
    sys.exit(main(sys.argv[1:]))
