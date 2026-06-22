"""
Reference utterances for the round-trip bench.

Short, clearly-spoken sentences (EN + PT-BR) of the kind a voice assistant handles:
greetings, numbers/amounts, scheduling, confirmations. Kept simple so a competent
STT backend should transcribe them at near-zero WER; persistent error highlights a
weak backend or a wiring bug.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceCase:
    text: str
    lang: str = "en"
    note: str = ""


CASES: list[VoiceCase] = [
    VoiceCase("hello, can you help me schedule an appointment?", "en", "greeting"),
    VoiceCase("my account balance is two hundred and fifty dollars", "en", "amount"),
    VoiceCase("please confirm the reservation for tomorrow at noon", "en", "scheduling"),
    VoiceCase("yes, that is correct, thank you very much", "en", "confirmation"),
    VoiceCase("the meeting is on Friday the twelfth of June", "en", "date"),
    VoiceCase("olá, gostaria de marcar uma consulta para meu cachorro", "pt", "greeting"),
    VoiceCase("o saldo da minha conta é de trezentos reais", "pt", "amount"),
    VoiceCase("por favor confirme a reserva para amanhã ao meio dia", "pt", "scheduling"),
]
