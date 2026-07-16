"""
cogno_vox.dia_server — OpenAI-compatible ``/v1/audio/speech`` wrapper for Dia TTS.

Dia (nari-labs, Apache-2.0) voices non-verbal cues from parenthetical tags —
``(laughs)``, ``(sighs)`` — which makes it the expressive tier in a TTS chain, but it
is a raw torch model, not an HTTP server. This module wraps ONE resident Dia
checkpoint behind the same ``POST /v1/audio/speech`` shape Kokoro/OpenAI speak, so
the existing :class:`~cogno_vox.synthesizer.OpenAICompatSynthesizer` (and therefore a
``TierConfig(provider="local", emotion_dialect="dia")``) works unchanged.

Run it as its own process (the model load takes ~40 s — it must stay resident)::

    pip install "cogno-vox[server]" torch safetensors soundfile
    pip install "git+https://github.com/nari-labs/dia.git@2811af1c"   # old config format
    python -m cogno_vox.dia_server --config config.json \
        --checkpoint Dia1.6-Portuguese-v1-merged.safetensors --port 8881

Notes
-----
* Community checkpoints (e.g. the pt-BR fine-tune) use the PRE-0626 config format —
  hence the pinned dia commit above — and ship ``.safetensors``, which that old lib's
  ``from_local`` cannot read; :class:`DiaEngine` loads the state dict itself.
* Dia expects a ``[S1]`` speaker prefix; the engine adds it when absent (a *server*
  concern — the emotion tags themselves are the chain's concern, see ``text_prep``).
* Heavy deps (torch / dia / safetensors / soundfile) are imported lazily so importing
  this module (or unit-testing the app with a fake engine) needs none of them.

(No ``from __future__ import annotations`` here on purpose: the request model is a
closure-local class, and stringified annotations would make FastAPI unable to resolve
it — the endpoint would silently degrade to a query parameter.)
"""

import io
import logging
import time
from typing import Any, Optional, Protocol, runtime_checkable

from cogno_vox.audio_utils import pcm_to_opus

log = logging.getLogger(__name__)

DIA_SAMPLE_RATE = 44100  # DAC codec output


@runtime_checkable
class SpeechEngine(Protocol):
    """What the app needs from an engine: text in, mono float PCM out."""

    sample_rate: int

    def generate(self, text: str) -> Any: ...  # numpy float array


class DiaEngine:
    """One resident Dia checkpoint (loaded once; ~40 s) behind :class:`SpeechEngine`."""

    sample_rate = DIA_SAMPLE_RATE

    def __init__(self, config_path: str, checkpoint_path: str, *,
                 compute_dtype: str = "float16", device: Optional[str] = None,
                 temperature: float = 1.2, cfg_scale: float = 3.0, top_p: float = 0.95,
                 max_tokens: int = 2048) -> None:  # pragma: no cover — heavy-deps loader
        # (torch/dia/safetensors; exercised by the live integration, not unit CI)
        import torch
        from dia.config import DiaConfig
        from dia.model import Dia
        from safetensors.torch import load_file

        dev = torch.device(device) if device else None
        config = DiaConfig.load(config_path)
        model = Dia(config, compute_dtype, dev)
        if checkpoint_path.endswith(".safetensors"):
            # the pre-0626 lib's from_local only knows torch.load/.pth
            state = load_file(checkpoint_path, device="cpu")
            model.model.load_state_dict(state)
        else:
            state = torch.load(checkpoint_path, map_location=model.device)
            model.model.load_state_dict(state)
        model.model.to(model.device)
        model.model.eval()
        model._load_dac_model()
        self._model = model
        self._gen = dict(temperature=temperature, cfg_scale=cfg_scale, top_p=top_p,
                         max_tokens=max_tokens)

    def generate(self, text: str) -> Any:
        if not text.lstrip().startswith("[S"):
            text = f"[S1] {text}"
        return self._model.generate(text, verbose=False, **self._gen)


def _to_wav(audio: Any, sample_rate: int) -> bytes:
    """Mono float PCM array → WAV bytes (stdlib wave — no soundfile at request time)."""
    import wave

    import numpy as np

    pcm = (np.clip(np.asarray(audio, dtype="float32"), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def create_app(engine: SpeechEngine) -> Any:
    """The FastAPI app over an injected engine (a fake in unit tests)."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel

    class SpeechRequest(BaseModel):
        input: str
        model: str = "dia"
        voice: str = ""              # single-voice checkpoint; accepted and ignored
        response_format: str = "wav"

    app = FastAPI(title="cogno-vox dia server")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "engine": type(engine).__name__}

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest) -> Response:
        text = (req.input or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty input")
        t0 = time.time()
        try:
            audio = engine.generate(text)
        except Exception as exc:  # noqa: BLE001 — surface as a 500 the chain fails over
            log.exception("event=dia_generate_failed")
            raise HTTPException(status_code=500, detail=str(exc))
        wav = _to_wav(audio, getattr(engine, "sample_rate", DIA_SAMPLE_RATE))
        log.info("event=dia_speech chars=%d ms=%.0f", len(text), (time.time() - t0) * 1000)
        if req.response_format == "opus":
            # WAV header is 44 bytes; pcm_to_opus wants raw s16le
            opus = pcm_to_opus(wav[44:], sample_rate=getattr(engine, "sample_rate",
                                                             DIA_SAMPLE_RATE))
            return Response(content=opus, media_type="audio/ogg")
        return Response(content=wav, media_type="audio/wav")

    return app


def main() -> None:  # pragma: no cover — thin CLI shell
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="OpenAI-compatible Dia TTS server")
    p.add_argument("--config", required=True, help="Dia config.json (pre-0626 format)")
    p.add_argument("--checkpoint", required=True, help=".safetensors/.pth checkpoint")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8881)
    p.add_argument("--device", default=None, help='e.g. "cuda" / "cpu" (default: auto)')
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    log.info("loading Dia checkpoint (this takes a while)…")
    engine = DiaEngine(args.config, args.checkpoint,
                       compute_dtype=args.dtype, device=args.device)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
