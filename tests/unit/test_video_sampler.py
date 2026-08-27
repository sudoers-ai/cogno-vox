"""Unit tests for cogno_vox.video_sampler."""

from pathlib import Path
from cogno_vox.video_sampler import extract_keyframes


def test_extract_keyframes_empty_input():
    frames = extract_keyframes(b"")
    assert frames == []


def test_extract_keyframes_invalid_video_bytes():
    frames = extract_keyframes(b"NOT_A_VIDEO_FILE")
    assert frames == []


def test_a_falta_da_DEPENDENCIA_diz_o_seu_nome(monkeypatch, caplog):
    """A ausência tem de ser AUDÍVEL, e a razão é que o silêncio é indistinguível do sucesso.

    `extract_keyframes` devolve `[]` sem a biblioteca — e `[]` é também o que devolve um vídeo
    sem mudanças de cena. **Um deploy que esqueceu `cogno-vox[vision]` processa vídeo, não regista
    nada, e responde como se o vídeo não tivesse nada dentro.** Degradar em silêncio não é
    gracioso: é mudo.
    """
    import logging

    from cogno_vox import video_sampler as vs

    monkeypatch.setattr(vs, "_HAS_CV2", False)
    monkeypatch.setattr(vs, "_WARNED_MISSING", False)
    with caplog.at_level(logging.WARNING):
        assert vs.extract_keyframes(b"nao-e-video") == []
    texto = "\n".join(r.getMessage() for r in caplog.records)
    assert "video_sampler_unavailable" in texto
    assert "cogno-vox[vision]" in texto, "o aviso tem de dizer COMO consertar, não só que falhou"


def test_o_aviso_sai_UMA_vez_e_nao_por_chamada(monkeypatch, caplog):
    """Um aviso por extracção treina o leitor a saltar a linha — e esta tem de ser legível no dia
    em que alguém pergunta porque é que o agente nunca vê nada num vídeo."""
    import logging

    from cogno_vox import video_sampler as vs

    monkeypatch.setattr(vs, "_HAS_CV2", False)
    monkeypatch.setattr(vs, "_WARNED_MISSING", False)
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            vs.extract_keyframes(b"x")
    avisos = [r for r in caplog.records if "video_sampler_unavailable" in r.getMessage()]
    assert len(avisos) == 1, f"saíram {len(avisos)} avisos em 5 chamadas"


def test_um_vídeo_VAZIO_nao_dispara_o_aviso_da_dependencia(monkeypatch, caplog):
    """O controlo do teste acima: prova que o aviso mede a DEPENDÊNCIA e não o caminho vazio.

    Sem isto, os dois testes passariam com o aviso a sair para qualquer `[]` — e a linha deixaria
    de distinguir as duas coisas que ela existe para distinguir.
    """
    import logging

    from cogno_vox import video_sampler as vs

    # `_HAS_CV2` é fixado a True DE PROPÓSITO: sem isso, numa máquina (ou CI) sem `cv2` este
    # teste exercita o caminho da dependência em falta em vez do caminho do vídeo vazio — e
    # passava a medir o contrário do que o nome diz.
    monkeypatch.setattr(vs, "_HAS_CV2", True)
    monkeypatch.setattr(vs, "_WARNED_MISSING", False)
    with caplog.at_level(logging.WARNING):
        assert vs.extract_keyframes(b"") == []
    assert not [r for r in caplog.records if "video_sampler_unavailable" in r.getMessage()]


def test_o_ficheiro_TEMPORARIO_nao_sobrevive_a_chamada(monkeypatch):
    """Apanhado por uma mutação sobrevivente: desligar a limpeza passava a suíte inteira.

    O ficheiro tem os BYTES DE VÍDEO DE UM CONTACTO no disco do sistema, em claro. `cv2` precisa
    de um caminho, portanto o ficheiro tem de existir — o que ele não pode é sobreviver à chamada.
    Sem este teste a limpeza era uma intenção escrita.
    """
    import tempfile as _tf

    from cogno_vox import video_sampler as vs

    criados: list = []
    original = _tf.NamedTemporaryFile

    def espia(*a, **kw):
        f = original(*a, **kw)
        criados.append(f.name)
        return f

    monkeypatch.setattr(vs.tempfile, "NamedTemporaryFile", espia)
    monkeypatch.setattr(vs, "_HAS_CV2", True)

    class _CapQuebrada:
        def isOpened(self):
            return False

        def release(self):
            pass

    # `raising=False`: sem o extra instalado o módulo não tem sequer o atributo `cv2`, e o
    # teste tem de correr na mesma — é o ambiente que este PR existe para tornar visível.
    monkeypatch.setattr(vs, "cv2", type("cv2", (), {"VideoCapture": lambda _p: _CapQuebrada()}),
                        raising=False)

    assert vs.extract_keyframes(b"bytes-de-um-contacto") == []
    assert criados, "o teste não observou nenhum ficheiro — não mediu nada"
    for nome in criados:
        assert not Path(nome).exists(), (
            f"{nome} sobreviveu à chamada — são bytes de vídeo de um contacto deixados no disco")
