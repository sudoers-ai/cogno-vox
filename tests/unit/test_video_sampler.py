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


def test_extract_keyframes_video_loop_and_scene_change_mocked(monkeypatch):
    """Testa o laço de amostragem de vídeo com cv2 mockado, garantindo cobertura 100% determinística.
    
    Verifica:
    1. Abertura do vídeo e contagem de fotogramas;
    2. Comparação de histograma e detecção de mudança de cena;
    3. Retorno de keyframes codificados em bytes.
    """
    from cogno_vox import video_sampler as vs

    class MockCap:
        def __init__(self, frames):
            self.frames = list(frames)
            self.idx = 0

        def isOpened(self):
            return self.idx < len(self.frames)

        def get(self, prop):
            return len(self.frames)

        def read(self):
            if self.idx < len(self.frames):
                frame = self.frames[self.idx]
                self.idx += 1
                return True, frame
            return False, None

        def release(self):
            pass

    class MockHist:
        def __init__(self, val):
            self.val = val

    class MockBuffer:
        def __init__(self, data):
            self.data = data

        def tobytes(self):
            return self.data

    class MockCv2:
        CAP_PROP_FRAME_COUNT = 4
        COLOR_BGR2HSV = 1
        NORM_MINMAX = 32
        HISTCMP_CORREL = 0

        def VideoCapture(self, path):
            return MockCap(["frame_black", "frame_black", "frame_white", "frame_white"])

        def cvtColor(self, frame, code):
            return frame

        def calcHist(self, images, channels, mask, histSize, ranges):
            return MockHist(1.0 if images[0] == "frame_black" else 0.0)

        def normalize(self, src, dst, alpha, beta, norm_type):
            pass

        def compareHist(self, h1, h2, method):
            return 1.0 if h1.val == h2.val else 0.0

        def imencode(self, ext, frame):
            return True, MockBuffer(f"encoded_{frame}".encode("utf-8"))

    monkeypatch.setattr(vs, "cv2", MockCv2(), raising=False)
    monkeypatch.setattr(vs, "np", type("np", (), {"ndarray": MockHist}), raising=False)
    monkeypatch.setattr(vs, "_HAS_CV2", True)

    keyframes = vs.extract_keyframes(b"fake_video_bytes", max_frames=8, scene_threshold=0.3)
    assert len(keyframes) == 2
    assert keyframes[0] == b"encoded_frame_black"
    assert keyframes[1] == b"encoded_frame_white"


def test_extract_keyframes_respeita_limite_max_frames(monkeypatch):
    """Garante que o limite max_frames é estritamente respeitado mesmo quando há mais mudanças de cena.
    
    Testado por sabotagem: remover `and len(keyframes) < max_frames` do laço faz este teste falhar.
    """
    from cogno_vox import video_sampler as vs

    class MockCap:
        def __init__(self, frames):
            self.frames = list(frames)
            self.idx = 0

        def isOpened(self):
            return self.idx < len(self.frames)

        def get(self, prop):
            return len(self.frames)

        def read(self):
            if self.idx < len(self.frames):
                frame = self.frames[self.idx]
                self.idx += 1
                return True, frame
            return False, None

        def release(self):
            pass

    class MockHist:
        def __init__(self, val):
            self.val = val

    class MockBuffer:
        def __init__(self, data):
            self.data = data

        def tobytes(self):
            return self.data

    class MockCv2Cap:
        CAP_PROP_FRAME_COUNT = 5
        COLOR_BGR2HSV = 1
        NORM_MINMAX = 32
        HISTCMP_CORREL = 0

        def VideoCapture(self, path):
            # 5 fotogramas completamente diferentes -> 5 mudanças de cena potenciais
            return MockCap(["f1", "f2", "f3", "f4", "f5"])

        def cvtColor(self, frame, code):
            return frame

        def calcHist(self, images, channels, mask, histSize, ranges):
            # Cada fotograma gera um histograma com valor distinto (1.0, 2.0, 3.0...)
            idx = int(images[0][1:])
            return MockHist(float(idx))

        def normalize(self, src, dst, alpha, beta, norm_type):
            pass

        def compareHist(self, h1, h2, method):
            # Sempre diferente se os valores forem distintos
            return 1.0 if h1.val == h2.val else 0.0

        def imencode(self, ext, frame):
            return True, MockBuffer(f"encoded_{frame}".encode("utf-8"))

    monkeypatch.setattr(vs, "cv2", MockCv2Cap(), raising=False)
    monkeypatch.setattr(vs, "np", type("np", (), {"ndarray": MockHist}), raising=False)
    monkeypatch.setattr(vs, "_HAS_CV2", True)

    # O vídeo tem 5 mudanças de cena, mas pedimos max_frames=2
    keyframes = vs.extract_keyframes(b"video_bytes_5_scenes", max_frames=2, scene_threshold=0.3)
    assert len(keyframes) == 2, f"deve cortar exatamente em 2 fotogramas, devolveu {len(keyframes)}"



def test_extract_keyframes_unopenable_video_returns_empty(monkeypatch):
    """Cobre a linha 85: quando cv2.VideoCapture falha em abrir o arquivo."""
    from cogno_vox import video_sampler as vs

    class MockCapUnopenable:
        def isOpened(self):
            return False

        def release(self):
            pass

    class MockCv2Unopenable:
        def VideoCapture(self, path):
            return MockCapUnopenable()

    monkeypatch.setattr(vs, "cv2", MockCv2Unopenable(), raising=False)
    monkeypatch.setattr(vs, "_HAS_CV2", True)

    assert vs.extract_keyframes(b"corrupt_video_bytes") == []


def test_extract_keyframes_fallback_total_frames_and_loop_break(monkeypatch):
    """Cobre linha 91 (total_frames <= 0) e linha 100 (if not ret: break)."""
    from cogno_vox import video_sampler as vs

    class MockCapZeroFrames:
        def __init__(self):
            self.read_count = 0

        def isOpened(self):
            return self.read_count < 2

        def get(self, prop):
            return 0  # total_frames <= 0 -> fallback para 100

        def read(self):
            self.read_count += 1
            if self.read_count == 1:
                return True, "frame_1"
            return False, None  # ret=False -> break

        def release(self):
            pass

    class MockBuffer:
        def tobytes(self):
            return b"frame_1_bytes"

    class MockCv2Zero:
        CAP_PROP_FRAME_COUNT = 0
        COLOR_BGR2HSV = 1
        NORM_MINMAX = 32
        HISTCMP_CORREL = 0

        def VideoCapture(self, path):
            return MockCapZeroFrames()

        def cvtColor(self, frame, code):
            return frame

        def calcHist(self, images, channels, mask, histSize, ranges):
            return "hist_1"

        def normalize(self, src, dst, alpha, beta, norm_type):
            pass

        def imencode(self, ext, frame):
            return True, MockBuffer()

    monkeypatch.setattr(vs, "cv2", MockCv2Zero(), raising=False)
    monkeypatch.setattr(vs, "np", type("np", (), {"ndarray": str}), raising=False)
    monkeypatch.setattr(vs, "_HAS_CV2", True)

    keyframes = vs.extract_keyframes(b"valid_bytes", max_frames=8)
    assert len(keyframes) == 1
    assert keyframes[0] == b"frame_1_bytes"


def test_extract_keyframes_exception_during_processing_returns_empty(monkeypatch, caplog):
    """Cobre as linhas 126-128: quando ocorre uma exceção durante o processamento do vídeo."""
    import logging
    from cogno_vox import video_sampler as vs

    class MockCapException:
        def isOpened(self):
            return True

        def get(self, prop):
            return 10

        def read(self):
            return True, "frame_err"

        def release(self):
            pass

    class MockCv2Exception:
        CAP_PROP_FRAME_COUNT = 0
        COLOR_BGR2HSV = 1

        def VideoCapture(self, path):
            return MockCapException()

        def cvtColor(self, frame, code):
            raise RuntimeError("Erro simulado de processamento de imagem")

    monkeypatch.setattr(vs, "cv2", MockCv2Exception(), raising=False)
    monkeypatch.setattr(vs, "_HAS_CV2", True)

    with caplog.at_level(logging.WARNING):
        assert vs.extract_keyframes(b"bytes_que_falham") == []

    logs = "\n".join(r.getMessage() for r in caplog.records)
    assert "video_sampler: keyframe extraction failed" in logs
    assert "Erro simulado de processamento de imagem" in logs




