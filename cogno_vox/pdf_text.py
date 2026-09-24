"""
cogno_vox.pdf_text — the TEXT LAYER of an UNTRUSTED PDF, read in a sandboxed process.

``PdfTextExtractor`` satisfies ``cogno_engram.documents.TextExtractor`` STRUCTURALLY — neither
library imports the other. What crosses between them is plain data: the bytes and three
ceilings in (``max_bytes``, ``max_pages``, ``timeout_s``), pages of text out (``pages[].number``,
``pages[].text``, and the bookmarks as ``outline[].level/.title/.page``), or an exception whose
``reason`` is one of five strings (:data:`PDF_REASONS`).

**The file is hostile until proven otherwise** — anybody with upload rights chose it — so:

1. **ceilings BEFORE reading**: over ``max_bytes`` is refused here, in this process, before a
   worker exists; the page count is read from the page tree and refused over ``max_pages``
   BEFORE any page is extracted (in the worker, the first thing it does with the document);
2. **a separate process with a deadline**: the parser is native code, a thread cannot be
   interrupted, and a file that spins or balloons must cost one process, not the caller's event
   loop. The worker is killed at ``timeout_s``; it also carries a CPU limit of its own, so it
   dies even if this process does;
3. **no network, no files**: inside the worker every descriptor above stdio is closed and the
   descriptor limit is 3 — no socket, no file, no pipe can be opened, by Python or by the
   native parser (see ``cogno_vox._pdf_worker``). No environment is inherited (``-E``, a minimal
   ``env``), and it runs in an empty temporary directory;
4. **text layer only**: forms are never initialised (no JavaScript), attachments never read,
   links never followed, nothing rendered.

A scanned PDF has no text layer and ends in ``no_text`` — OCR is not included.

Needs ``pypdfium2`` (``pip install "cogno-vox[pdf]"`` — the extra carries nothing else; the
``vision`` extra's opencv and numpy are not needed for this).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cogno_vox.ports import VoxError

PDF_MEDIA_TYPE = "application/pdf"

# The extractor's closed alphabet — the SAME five strings as
# `cogno_engram.documents.EXTRACTOR_REASONS` (pinned by literal in the tests; neither library
# imports the other).
REASON_NO_TEXT = "no_text"          # no text layer — a scanned PDF (OCR is not included)
REASON_OVER_LIMIT = "over_limit"    # bytes, pages, extracted text or memory over a ceiling
REASON_ENCRYPTED = "encrypted"      # password-protected
REASON_INVALID = "invalid"          # not a PDF, corrupt, or the parser failed on it
REASON_TIMEOUT = "timeout"          # the worker did not finish in time (killed)
PDF_REASONS: frozenset = frozenset({REASON_NO_TEXT, REASON_OVER_LIMIT, REASON_ENCRYPTED,
                                    REASON_INVALID, REASON_TIMEOUT})

#: Memory the worker may map (address space). A PDF that makes the parser balloon past it ends
#: in ``over_limit``.
DEFAULT_MEMORY_BYTES = 1024 * 1024 * 1024
#: Text the worker may return, in characters — a 300-page document of dense prose is ~1 M.
DEFAULT_MAX_TEXT_CHARS = 20_000_000
#: Where a PDF header may start (the spec allows leading garbage; 1 KiB is what readers accept).
_HEADER_WINDOW = 1024

_WORKER_SOURCE = (Path(__file__).with_name("_pdf_worker.py")).read_text(encoding="utf-8")


class PdfExtractionError(VoxError):
    """An extraction that ended for a reason the uploader must see. ``reason`` is one of
    :data:`PDF_REASONS`; the message never carries a byte of the file."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason if reason in PDF_REASONS else REASON_INVALID
        self.detail = detail
        super().__init__(f"{self.reason}: {detail}" if detail else self.reason)


@dataclass(frozen=True)
class PdfPage:
    number: int          # 1-based
    text: str


@dataclass(frozen=True)
class PdfBookmark:
    level: int           # 1-based depth
    title: str
    page: int            # 1-based


@dataclass(frozen=True)
class PdfText:
    pages: tuple = ()
    outline: tuple = field(default=())


def pdf_support_available() -> bool:
    """Whether ``pypdfium2`` (the ``pdf`` extra) is installed."""
    return importlib.util.find_spec("pypdfium2") is not None


def _worker_env() -> dict:
    """The worker's WHOLE environment: nothing inherited — no proxy, no credential, no
    ``PYTHONPATH``. ``HOME`` stays only so an interpreter that finds its packages in the user
    site still finds them."""
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    if os.environ.get("HOME"):
        env["HOME"] = os.environ["HOME"]
    return env


class PdfTextExtractor:
    """A ``TextExtractor`` for PDF — see the module docstring for the guarantees.

    ``python`` is the interpreter the worker runs under (this one by default); it must be able
    to import ``pypdfium2``. The constructor fails LOUDLY when it cannot, so a host wires a PDF
    extractor that works or learns at boot that it has none — never an extractor that turns
    every upload into ``invalid``.
    """

    media_types = frozenset({PDF_MEDIA_TYPE})

    def __init__(self, *, memory_bytes: int = DEFAULT_MEMORY_BYTES,
                 max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
                 python: Optional[str] = None) -> None:
        if python is None and not pdf_support_available():
            raise ImportError('PDF text extraction needs pypdfium2: pip install "cogno-vox[pdf]"')
        self.memory_bytes = int(memory_bytes)
        self.max_text_chars = int(max_text_chars)
        self.python = python or sys.executable
        #: The pid of the last worker — so a test can prove it is gone after a timeout.
        self.last_pid: Optional[int] = None

    async def extract(self, data: bytes, *, media_type: str, max_bytes: int, max_pages: int,
                      timeout_s: float) -> PdfText:
        if media_type != PDF_MEDIA_TYPE:
            raise PdfExtractionError(REASON_INVALID, f"not a PDF media type: {media_type!r}")
        if not isinstance(data, (bytes, bytearray)):
            raise PdfExtractionError(REASON_INVALID, "data must be bytes")
        if len(data) > int(max_bytes):
            raise PdfExtractionError(REASON_OVER_LIMIT, f"{len(data)} bytes")
        if b"%PDF-" not in bytes(data[:_HEADER_WINDOW]):
            raise PdfExtractionError(REASON_INVALID, "no PDF header")
        raw = await self._run("extract", bytes(data), max_pages=int(max_pages),
                              timeout_s=float(timeout_s))
        if "error" in raw:
            raise PdfExtractionError(str(raw["error"]))
        try:
            pages = tuple(PdfPage(number=int(p["number"]), text=str(p["text"]))
                          for p in raw["pages"])
            outline = tuple(PdfBookmark(level=int(b["level"]), title=str(b["title"]),
                                        page=int(b["page"])) for b in raw.get("outline", ()))
        except (KeyError, TypeError, ValueError) as exc:
            raise PdfExtractionError(REASON_INVALID, "worker answered an unreadable shape") \
                from exc
        return PdfText(pages=pages, outline=outline)

    async def selfcheck(self, *, timeout_s: float = 30.0) -> dict:
        """What the worker's sandbox refuses, measured from inside it:
        ``{"socket", "raw_socket", "open", "memory"}`` — anything reading ``allowed`` is a door
        left open. Cheap; a host may run it once at boot."""
        return await self._run("selfcheck", b"", max_pages=0, timeout_s=timeout_s)

    async def _run(self, mode: str, data: bytes, *, max_pages: int, timeout_s: float) -> dict:
        cpu_seconds = max(1, math.ceil(timeout_s)) + 1
        with tempfile.TemporaryDirectory(prefix="cogno-vox-pdf-") as cwd:
            proc = await asyncio.create_subprocess_exec(
                self.python, "-E", "-c", _WORKER_SOURCE, mode, str(max_pages),
                str(self.memory_bytes), str(cpu_seconds), str(self.max_text_chars),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, cwd=cwd, env=_worker_env(),
                start_new_session=True)
            self.last_pid = proc.pid
            try:
                out, _ = await asyncio.wait_for(proc.communicate(data), timeout=timeout_s)
            except asyncio.TimeoutError:
                await self._kill(proc)
                raise PdfExtractionError(REASON_TIMEOUT, f"over {timeout_s}s") from None
            except BaseException:
                await self._kill(proc)
                raise
        if proc.returncode != 0:
            # Killed by its own CPU limit (SIGXCPU) or by a signal: the deadline, reached from
            # inside. Any other failure to finish is the parser failing on the file.
            reason = REASON_TIMEOUT if (proc.returncode or 0) < 0 else REASON_INVALID
            raise PdfExtractionError(reason, f"worker exited {proc.returncode}")
        try:
            answer = json.loads(out.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PdfExtractionError(REASON_INVALID, "worker answered no JSON") from exc
        if not isinstance(answer, dict):
            raise PdfExtractionError(REASON_INVALID, "worker answered no object")
        return answer

    @staticmethod
    async def _kill(proc: "asyncio.subprocess.Process") -> None:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
