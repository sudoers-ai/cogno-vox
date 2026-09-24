"""``PdfTextExtractor`` — an UNTRUSTED PDF, read in a sandboxed process (F2.4, risk 5).

Each guarantee is tested where it lives, with its CONTROL:

* the byte ceiling and the header check refuse BEFORE a worker exists (the control lets one be
  spawned);
* the page ceiling refuses BEFORE any page is read (in-process, a document whose pages explode
  when touched; the control reads them);
* the worker cannot open a socket — at the KERNEL level, under the Python patch — nor a file,
  nor balloon past its memory (the control: the same calls work in this process);
* a worker that does not finish is killed and gone;
* the answer has the shape ``cogno_engram.documents.read_extracted`` reads, and the reasons are
  the five strings of that contract.

Real subprocesses — pypdfium2 is the ``pdf`` extra (CI installs it). Invented content only.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys

import pytest

pdfium = pytest.importorskip("pypdfium2")

from cogno_vox import pdf_text  # noqa: E402
from cogno_vox import _pdf_worker as worker  # noqa: E402
from cogno_vox.pdf_text import (  # noqa: E402
    PDF_MEDIA_TYPE,
    PDF_REASONS,
    PdfExtractionError,
    PdfTextExtractor,
)
from tests.unit.pdf_fixtures import ENCRYPTED_PDF, OUTLINED_PDF  # noqa: E402

LIMITS = dict(media_type=PDF_MEDIA_TYPE, max_bytes=10 * 1024 * 1024, max_pages=300, timeout_s=30.0)


def extractor(**kw) -> PdfTextExtractor:
    return PdfTextExtractor(**kw)


async def reason_of(coro) -> str:
    with pytest.raises(PdfExtractionError) as err:
        await coro
    return err.value.reason


def test_the_reasons_are_the_five_of_the_engram_contract():
    # `cogno_engram.documents.EXTRACTOR_REASONS` — by literal: neither library imports the other.
    assert PDF_REASONS == {"no_text", "over_limit", "encrypted", "invalid", "timeout"}
    assert {worker.NO_TEXT, worker.OVER_LIMIT, worker.ENCRYPTED, worker.INVALID,
            worker.TIMEOUT} == PDF_REASONS


# ── the happy path, through the real worker ─────────────────────────────────────────────

async def test_pages_come_back_numbered_with_their_text():
    data = worker.tiny_pdf(["Horario de sabado: 8h as 12h.", "", "Pagina tres."])
    got = await extractor().extract(data, **LIMITS)
    assert [(p.number, p.text) for p in got.pages] == [
        (1, "Horario de sabado: 8h as 12h."), (2, ""), (3, "Pagina tres.")]
    assert got.outline == ()


async def test_the_bookmarks_come_back_as_the_outline():
    got = await extractor().extract(OUTLINED_PDF, **LIMITS)
    assert [p.text for p in got.pages] == ["Introducao ao regulamento.", "Sabado: 8h as 12h.",
                                           "Multas por atraso."]
    assert [(b.level, b.title, b.page) for b in got.outline] == [
        (1, "Funcionamento", 2), (2, "Sabado", 2), (1, "Penalidades", 3)]


async def test_the_answer_has_the_shape_the_engram_reads():
    """``read_extracted`` reads ``pages[].number/.text`` and ``outline[].level/.title/.page`` by
    attribute — the structural half of the contract."""
    got = await extractor().extract(OUTLINED_PDF, **LIMITS)
    page, mark = got.pages[0], got.outline[0]
    assert isinstance(page.number, int) and isinstance(page.text, str)
    assert isinstance(mark.level, int) and isinstance(mark.title, str) and isinstance(mark.page, int)
    assert PdfTextExtractor.media_types == frozenset({"application/pdf"})


# ── ceilings BEFORE reading ──────────────────────────────────────────────────────────────

async def test_over_the_byte_ceiling_is_refused_before_a_worker_exists(monkeypatch):
    spawned: list = []
    real = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        spawned.append(args)
        return await real(*args, **kwargs)

    monkeypatch.setattr(pdf_text.asyncio, "create_subprocess_exec", spy)
    data = worker.tiny_pdf(["x"])
    assert await reason_of(extractor().extract(data, **{**LIMITS, "max_bytes": len(data) - 1})) \
        == "over_limit"
    assert spawned == []
    await extractor().extract(data, **{**LIMITS, "max_bytes": len(data)})     # CONTROL
    assert len(spawned) == 1


async def test_no_pdf_header_is_invalid_before_a_worker_exists(monkeypatch):
    async def never(*a, **k):
        raise AssertionError("a worker was spawned for a file with no PDF header")

    monkeypatch.setattr(pdf_text.asyncio, "create_subprocess_exec", never)
    assert await reason_of(extractor().extract(b"<html>not a pdf</html>", **LIMITS)) == "invalid"
    assert await reason_of(extractor().extract(b"%PDF-1.4", **{**LIMITS, "media_type": "text/html"})) \
        == "invalid"
    assert await reason_of(extractor().extract("text", **LIMITS)) == "invalid"   # type: ignore[arg-type]


async def test_over_the_page_ceiling_is_refused_by_the_worker():
    data = worker.tiny_pdf(["a", "b", "c"])
    assert await reason_of(extractor().extract(data, **{**LIMITS, "max_pages": 2})) == "over_limit"
    got = await extractor().extract(data, **{**LIMITS, "max_pages": 3})          # CONTROL
    assert len(got.pages) == 3


def test_the_page_ceiling_is_checked_before_ANY_page_is_read(monkeypatch):
    touched: list[int] = []
    real_getitem = pdfium.PdfDocument.__getitem__

    def spy(self, index):
        touched.append(index)
        return real_getitem(self, index)

    monkeypatch.setattr(pdfium.PdfDocument, "__getitem__", spy)
    data = worker.tiny_pdf(["a", "b", "c"])
    assert worker.read_text(pdfium, data, max_pages=2, max_chars=10**6) == {"error": "over_limit"}
    assert touched == []
    worker.read_text(pdfium, data, max_pages=3, max_chars=10**6)                 # CONTROL
    assert touched == [0, 1, 2]


async def test_more_text_than_the_ceiling_is_over_limit():
    data = worker.tiny_pdf(["x" * 60, "y" * 60])
    assert await reason_of(extractor(max_text_chars=100).extract(data, **LIMITS)) == "over_limit"
    assert len((await extractor(max_text_chars=120).extract(data, **LIMITS)).pages) == 2


# ── the file's own failures ─────────────────────────────────────────────────────────────

async def test_a_protected_pdf_is_encrypted():
    assert await reason_of(extractor().extract(ENCRYPTED_PDF, **LIMITS)) == "encrypted"


async def test_a_pdf_without_a_text_layer_is_no_text():
    assert await reason_of(extractor().extract(worker.tiny_pdf(["", ""]), **LIMITS)) == "no_text"


async def test_garbage_behind_a_pdf_header_is_invalid():
    assert await reason_of(extractor().extract(b"%PDF-1.7\n" + os.urandom(2048), **LIMITS)) \
        == "invalid"


def test_the_in_process_reader_covers_every_answer():
    assert worker.read_text(pdfium, ENCRYPTED_PDF, max_pages=9, max_chars=9) == {"error": "encrypted"}
    assert worker.read_text(pdfium, b"%PDF-1.4 no", max_pages=9, max_chars=9) == {"error": "invalid"}
    assert worker.read_text(pdfium, worker.tiny_pdf([""]), max_pages=9, max_chars=9) \
        == {"error": "no_text"}
    ok = worker.read_text(pdfium, OUTLINED_PDF, max_pages=9, max_chars=10**6)
    assert len(ok["pages"]) == 3 and len(ok["outline"]) == 3
    worker.warm_up(pdfium)


def test_the_outline_is_bounded(monkeypatch):
    monkeypatch.setattr(worker, "MAX_OUTLINE", 1)
    ok = worker.read_text(pdfium, OUTLINED_PDF, max_pages=9, max_chars=10**6)
    assert [b["title"] for b in ok["outline"]] == ["Funcionamento"]


# ── the sandbox, measured from inside the worker ─────────────────────────────────────────

async def test_the_worker_can_open_no_socket_no_file_and_cannot_balloon():
    shut = await extractor(memory_bytes=512 * 1024 * 1024).selfcheck()
    assert shut == {"socket": "PermissionError",            # the Python patch
                    "raw_socket": "OSError:24",             # the KERNEL: EMFILE, no descriptor
                    "open": "OSError",
                    "memory": "MemoryError"}
    # CONTROL — the very calls the worker is refused succeed in this process.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()
    fd = os.open(os.devnull, os.O_RDONLY)
    os.close(fd)


async def test_the_worker_inherits_no_environment_and_runs_isolated(monkeypatch):
    seen: list = []
    real = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return await real(*args, **kwargs)

    monkeypatch.setenv("COGNO_TEST_SECRET", "must-not-cross")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:1")
    monkeypatch.setattr(pdf_text.asyncio, "create_subprocess_exec", spy)
    await extractor().extract(worker.tiny_pdf(["x"]), **LIMITS)
    [(args, kwargs)] = seen
    assert args[1:3] == ("-E", "-c")
    assert set(kwargs["env"]) <= {"PATH", "LANG", "LC_ALL", "HOME"}
    assert kwargs["cwd"] != os.getcwd() and kwargs["start_new_session"] is True


# ── a worker that does not finish ───────────────────────────────────────────────────────

async def test_a_worker_past_its_deadline_is_killed_and_gone():
    ex = extractor()
    assert await reason_of(ex.extract(worker.tiny_pdf(["x"]), **{**LIMITS, "timeout_s": 0.001})) \
        == "timeout"
    assert ex.last_pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(ex.last_pid, 0)


async def test_a_worker_killed_by_a_signal_is_timeout_and_a_crash_is_invalid(tmp_path):
    killed = tmp_path / "killed.sh"
    killed.write_text("#!/bin/sh\nkill -9 $$\n")
    killed.chmod(0o755)
    data = worker.tiny_pdf(["x"])
    assert await reason_of(extractor(python=str(killed)).extract(data, **LIMITS)) == "timeout"
    assert await reason_of(extractor(python="/bin/false").extract(data, **LIMITS)) == "invalid"
    assert await reason_of(extractor(python="/bin/echo").extract(data, **LIMITS)) == "invalid"


async def test_a_worker_answering_the_wrong_shape_is_invalid(tmp_path):
    for body in ('[]', '{"pages": [{"number": "x"}]}', '{"error": "made-up"}'):
        fake = tmp_path / "fake.sh"
        fake.write_text(f"#!/bin/sh\ncat >/dev/null\nprintf '%s' '{body}'\n")
        fake.chmod(0o755)
        assert await reason_of(extractor(python=str(fake)).extract(worker.tiny_pdf(["x"]),
                                                                   **LIMITS)) == "invalid", body


async def test_a_cancelled_extraction_kills_its_worker(tmp_path):
    slow = tmp_path / "slow.sh"
    slow.write_text("#!/bin/sh\nexec sleep 30\n")
    slow.chmod(0o755)
    ex = extractor(python=str(slow))
    task = asyncio.create_task(ex.extract(worker.tiny_pdf(["x"]), **LIMITS))
    for _ in range(100):
        await asyncio.sleep(0.02)
        if ex.last_pid is not None:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(ex.last_pid, 0)


def test_without_pypdfium2_the_constructor_fails_loudly(monkeypatch):
    monkeypatch.setattr(pdf_text.importlib.util, "find_spec", lambda name: None)
    assert pdf_text.pdf_support_available() is False
    with pytest.raises(ImportError):
        PdfTextExtractor()
    PdfTextExtractor(python=sys.executable)            # an explicit interpreter is the caller's


def test_an_unknown_reason_is_invalid():
    assert PdfExtractionError("made-up").reason == "invalid"
    assert PdfExtractionError("timeout", "x").reason == "timeout"
