# Logging — convenção desta lib

Esta biblioteca **emite** logs; o **host configura** (handlers, formato, nível,
contexto de tenant). Regras:

1. Use `logging.getLogger(__name__)` no topo do módulo. Nada de handlers,
   formatters, `basicConfig` ou um `get_logger` próprio.
2. Mensagem = só o fato de domínio, em `key=value`, sempre lazy:
   `logger.info("event=transcribe_done provider=%s audio_ms=%d", p, ms)`.
   NÃO coloque tenant_id / timestamp / channel na mensagem — o host injeta
   via contextvars + Filter no root logger (carimbado em todo LogRecord).
3. Níveis:
   - **ERROR**  → nunca aqui; erro fatal vira exceção e propaga (host loga ERROR).
   - **WARNING**→ condição recuperada/tratada (fallback, parse coercion, verify falho).
   - **INFO**   → marco caro e raro; NÃO happy-path por request.
   - **DEBUG**  → trace de fidelidade total (chunking/parâmetros). DEV-ONLY,
                  jamais ligado em produção multi-tenant. Redija secrets (apikey).
4. Controle de nível é por pacote: `logging.getLogger("cogno_vox").setLevel(...)`.

O host anexa o handler (TenantFilter + JsonFormatter) ao root logger real;
veja `cogno/core/logging.py` no host como referência.

## Nota específica do cogno-vox

- **WARNING** quando o provedor/motor local falha e aciona o fallback na nuvem
  (STT: provedor local → cloud; TTS: Kokoro → cloud).
- **INFO** no marco concluído (transcrição: duração/provedor; síntese: bytes).
- **DEBUG** em chunking de texto e parâmetros de requisição. O áudio e a
  transcrição são conteúdo de usuário → DEBUG apenas (dev-only).
