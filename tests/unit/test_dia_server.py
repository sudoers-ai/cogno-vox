"""Dia wrapper server — the OpenAI-compat contract over a fake engine (no torch/dia)."""

from __future__ import annotations

import wave
from io import BytesIO

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cogno_vox.dia_server import create_app  # noqa: E402


class FakeEngine:
    """Records the text and returns 100 ms of silence at Dia's rate."""

    sample_rate = 44100

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.texts: list[str] = []

    def generate(self, text: str):
        if self.fail:
            raise RuntimeError("boom")
        self.texts.append(text)
        return [0.0] * (self.sample_rate // 10)


def _client(engine=None):
    eng = engine if engine is not None else FakeEngine()
    return TestClient(create_app(eng)), eng


def test_speech_returns_wav():
    client, eng = _client()
    r = client.post("/v1/audio/speech",
                    json={"model": "dia", "input": "Olá! (laughs) Tudo bem?"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    with wave.open(BytesIO(r.content)) as w:
        assert w.getframerate() == 44100 and w.getnchannels() == 1
        assert w.getnframes() == 4410           # os 100 ms do fake


def test_openai_payload_shape_is_accepted():
    # o formato exato que o OpenAICompatSynthesizer do vox envia
    client, eng = _client()
    r = client.post("/v1/audio/speech", json={
        "model": "dia-ptbr", "input": "Oi.", "voice": "alloy", "response_format": "wav"})
    assert r.status_code == 200


def test_s1_prefix_is_engine_concern_not_added_twice():
    # o DiaEngine real adiciona [S1]; um texto já prefixado passa intacto pelo app
    client, eng = _client()
    client.post("/v1/audio/speech", json={"input": "[S1] Olá."})
    assert eng.texts == ["[S1] Olá."]


def test_empty_input_is_400():
    client, _ = _client()
    assert client.post("/v1/audio/speech", json={"input": "  "}).status_code == 400


def test_engine_error_is_500_so_chain_fails_over():
    client, _ = _client(FakeEngine(fail=True))
    assert client.post("/v1/audio/speech", json={"input": "Oi."}).status_code == 500


def test_health():
    client, _ = _client()
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["engine"] == "FakeEngine"


def test_dia_engine_prefixes_s1(monkeypatch):
    # unit da regra de prefixo sem carregar o modelo: instância sem __init__
    from cogno_vox.dia_server import DiaEngine

    captured = {}

    class FakeModel:
        def generate(self, text, **kw):
            captured["text"] = text
            return [0.0]

    eng = DiaEngine.__new__(DiaEngine)
    eng._model = FakeModel()
    eng._gen = {}
    eng.generate("Olá! Tudo bem?")
    assert captured["text"] == "[S1] Olá! Tudo bem?"
    eng.generate("[S1] Já prefixado.")
    assert captured["text"] == "[S1] Já prefixado."
