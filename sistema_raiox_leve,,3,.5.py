import math
import re
import sys
import time
import unicodedata
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

# MediaPipe é opcional: sem ele o programa roda só com o filtro do OpenCV.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
PASTA_BASE = Path(__file__).resolve().parent / "exames"
FORMATO_DATA_HORA = "%d/%m/%Y %H:%M"
FORMATO_NASCIMENTO = "%d/%m/%Y"
INTERVALO_VIDEO_MS = 20
TEMPO_CONGELADO_MS = 1500

# Modelo do Pose Landmarker (baixado automaticamente na primeira execução)
CAMINHO_MODELO = Path(__file__).resolve().parent / "modelos" / "pose_landmarker_lite.task"
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
VISIBILIDADE_MIN = 0.5

# Cores do esqueleto (BGR)
COR_OSSO = (235, 235, 235)

# Índices dos landmarks do MediaPipe Pose (lado E = esquerda anatômica do paciente)
IDX = {
    "E": dict(ombro=11, cotovelo=13, punho=15, mindinho=17, indicador=19, polegar=21,
              quadril=23, joelho=25, tornozelo=27, calcanhar=29, ponta_pe=31),
    "D": dict(ombro=12, cotovelo=14, punho=16, mindinho=18, indicador=20, polegar=22,
              quadril=24, joelho=26, tornozelo=28, calcanhar=30, ponta_pe=32),
}
IDX_NARIZ = 0
IDX_OLHO = {"E": 2, "D": 5}
IDX_ORELHA = {"E": 7, "D": 8}
IDX_BOCA = {"E": 9, "D": 10}

PARTES_LATERAIS = {"clavicula", "ombro", "braco", "antebraco", "mao", "coxa", "perna", "pe"}

LISTA_EXAMES = [
    # Torax e Aparelho Respiratorio
    "TORAX PA",
    "TORAX PERFIL",
    "COSTELAS / HEMITORAX",
    "ESTERNO",
    "CORACAO E GRANDES VASOS",

    # Coluna Vertebral
    "COLUNA CERVICAL",
    "COLUNA CERVICAL DINAMICA",
    "COLUNA DORSAL (TORACICA)",
    "COLUNA LOMBAR",
    "COLUNA LOMBO-SACRA",
    "COLUNA LOMBAR DINAMICA",
    "REGIAO SACRO-COXIGEA",
    "COLUNA TOTAL (ESCOLIOSE)",

    # Cranio e Face
    "CRANIO",
    "SEIOS DA FACE",
    "CAVUM / ADENOIDES",
    "OSSOS DA FACE",
    "MAXILAR",
    "MANDIBULA",
    "ARCADA ZIGOMATICA",
    "ARTICULACAO TEMPOROMANDIBULAR (ATM)",
    "SELA TURCICA",
    "ORBITAS",
    "MASTOIDES",

    # Membros Superiores
    "MEMBRO SUPERIOR ESQ",
    "MEMBRO SUPERIOR DIR",
    "OMBRO ESQ",
    "OMBRO DIR",
    "ARTICULACAO ACROMIOCLAVICULAR ESQ",
    "ARTICULACAO ACROMIOCLAVICULAR DIR",
    "ESCAPULA ESQ",
    "ESCAPULA DIR",
    "CLAVICULA ESQ",
    "CLAVICULA DIR",
    "BRACO ESQ",
    "BRACO DIR",
    "COTOVELO ESQ",
    "COTOVELO DIR",
    "ANTEBRACO ESQ",
    "ANTEBRACO DIR",
    "PUNHO ESQ",
    "PUNHO DIR",
    "MAO ESQ",
    "MAO DIR",
    "DEDOS DA MAO ESQ",
    "DEDOS DA MAO DIR",
    "MAOS E PUNHOS (IDADE OSSEA)",

    # Membros Inferiores
    "MEMBRO INFERIOR ESQ",
    "MEMBRO INFERIOR DIR",
    "BACIA",
    "ARTICULACAO COXOFEMORAL (QUADRIL) ESQ",
    "ARTICULACAO COXOFEMORAL (QUADRIL) DIR",
    "ARTICULACOES SACROILIACAS",
    "COXA (FEMUR) ESQ",
    "COXA (FEMUR) DIR",
    "JOELHO ESQ",
    "JOELHO DIR",
    "PATELA ESQ",
    "PATELA DIR",
    "PERNA (TIBIA E FIBULA) ESQ",
    "PERNA (TIBIA E FIBULA) DIR",
    "TORNOZELO ESQ",
    "TORNOZELO DIR",
    "PE ESQ",
    "PE DIR",
    "CALCANEO ESQ",
    "CALCANEO DIR",
    "ESCANOMETRIA MMII",

    # Abdomen e Pelve
    "ABDOMEN SIMPLES",
    "ABDOMEN AGUDO",
    "VIAS URINARIAS / RENAL",
]


# ----------------------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------------------
def remover_acentos(texto):
    """Remove acentos (o cv2.putText não desenha caracteres acentuados)."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def limpar_nome(texto):
    """Gera um nome seguro para usar em pasta/arquivo."""
    texto = remover_acentos(texto)
    texto = re.sub(r"[^\w\- ]", "", texto).strip().replace(" ", "_")
    return texto or "paciente"


def data_nascimento_valida(texto):
    """Campo vazio é aceito; se preenchido, precisa estar em dd/mm/aaaa."""
    if not texto.strip():
        return True
    try:
        data = datetime.strptime(texto.strip(), FORMATO_NASCIMENTO)
        return data <= datetime.now()
    except ValueError:
        return False


def garantir_modelo():
    """Baixa o modelo do Pose Landmarker se ainda não existir. Retorna True se estiver disponível."""
    if CAMINHO_MODELO.exists() and CAMINHO_MODELO.stat().st_size > 0:
        return True
    try:
        CAMINHO_MODELO.parent.mkdir(parents=True, exist_ok=True)
        temporario = CAMINHO_MODELO.with_suffix(".tmp")
        urllib.request.urlretrieve(URL_MODELO, temporario)
        temporario.replace(CAMINHO_MODELO)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# Geometria e desenho do esqueleto
# ----------------------------------------------------------------------
def distancia(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def media(a, b):
    return lerp(a, b, 0.5)


def inteiro(p):
    return (int(round(p[0])), int(round(p[1])))


def ponto(pontos, i):
    """Retorna (x, y) em pixels se o landmark estiver visível o bastante; senão None."""
    x, y, vis = pontos[i]
    return (x, y) if vis >= VISIBILIDADE_MIN else None


def partes_do_exame(local):
    """
    Define quais estruturas ósseas desenhar conforme o exame escolhido.
    Retorna {parte: [lados]}; partes centrais usam lados = [None].
    """
    nome = remover_acentos(local).upper()
    lados = ["E"] if " ESQ" in nome else ["D"] if " DIR" in nome else ["E", "D"]
    partes = {}

    def add(*itens):
        for item in itens:
            partes[item] = lados if item in PARTES_LATERAIS else [None]

    if "MEMBRO SUPERIOR" in nome:
        add("clavicula", "ombro", "braco", "antebraco", "mao")
    elif "MAOS E PUNHOS" in nome:
        partes.update({"antebraco": ["E", "D"], "mao": ["E", "D"]})
    elif "ANTEBRACO" in nome:
        add("antebraco")
    elif "COTOVELO" in nome:
        add("braco", "antebraco")
    elif "BRACO" in nome:
        add("braco")
    elif "PUNHO" in nome:
        add("antebraco", "mao")
    elif "MAO" in nome or "DEDOS" in nome:
        add("mao")
    elif "ACROMIO" in nome or "ESCAPULA" in nome or "CLAVICULA" in nome:
        add("clavicula", "ombro")
    elif "OMBRO" in nome:
        add("clavicula", "ombro", "braco")
    elif "MEMBRO INFERIOR" in nome:
        add("coxa", "perna", "pe")
        add("pelve")
    elif "ESCANOMETRIA" in nome:
        partes.update({"coxa": ["E", "D"], "perna": ["E", "D"], "pe": ["E", "D"], "pelve": [None]})
    elif "COXOFEMORAL" in nome:
        add("pelve", "coxa")
    elif "COXA" in nome:
        add("coxa")
    elif "JOELHO" in nome or "PATELA" in nome:
        add("coxa", "perna")
    elif "PERNA" in nome:
        add("perna")
    elif "TORNOZELO" in nome:
        add("perna", "pe")
    elif "CALCANEO" in nome or nome.startswith("PE "):
        add("pe")
    elif "BACIA" in nome or "SACROILIACA" in nome:
        add("pelve")
    elif "SACRO-COXIGEA" in nome:
        add("pelve", "coluna_lombar")
    elif "COLUNA TOTAL" in nome:
        add("coluna_cervical", "coluna_dorsal", "coluna_lombar", "pelve")
    elif "CERVICAL" in nome:
        add("coluna_cervical")
    elif "DORSAL" in nome:
        add("coluna_dorsal", "costelas")
    elif "LOMBO" in nome or "LOMBAR" in nome:
        add("coluna_lombar", "pelve")
    elif "ABDOMEN" in nome or "VIAS URINARIAS" in nome:
        add("coluna_lombar", "pelve")
    elif any(p in nome for p in ("TORAX", "COSTELAS", "ESTERNO", "CORACAO")):
        add("costelas", "esterno", "clavicula", "coluna_dorsal")
    elif any(p in nome for p in ("CRANIO", "FACE", "SEIOS", "CAVUM", "MAXILAR", "MANDIBULA",
                                 "ZIGOMATICA", "TEMPOROMANDIBULAR", "SELA", "ORBITAS", "MASTOIDES")):
        add("cranio")
    else:  # exame desconhecido: mostra o corpo todo
        add("cranio", "costelas", "esterno", "clavicula", "coluna_cervical", "coluna_dorsal",
            "coluna_lombar", "pelve", "ombro", "braco", "antebraco", "mao", "coxa", "perna", "pe")
    return partes


def desenhar_esqueleto(imagem, pontos, partes):
    """Desenha os ossos sobre a imagem de raio-x e devolve uma nova imagem."""
    camada = np.zeros_like(imagem)

    # Escala baseada na largura dos ombros (ou valor padrão se não estiverem visíveis)
    oe, od = ponto(pontos, IDX["E"]["ombro"]), ponto(pontos, IDX["D"]["ombro"])
    escala = max(distancia(oe, od), 40.0) if (oe and od) else 160.0
    esp = max(2, int(escala * 0.035))

    def osso(p1, p2, espessura=None):
        if p1 is None or p2 is None:
            return
        e = espessura or esp
        cv2.line(camada, inteiro(p1), inteiro(p2), COR_OSSO, e, cv2.LINE_AA)
        cv2.circle(camada, inteiro(p1), e, COR_OSSO, -1, cv2.LINE_AA)
        cv2.circle(camada, inteiro(p2), e, COR_OSSO, -1, cv2.LINE_AA)

    def osso_duplo(p1, p2, afastamento):
        """Dois ossos paralelos (rádio/ulna, tíbia/fíbula)."""
        if p1 is None or p2 is None:
            return
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        norma = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norma * afastamento, dx / norma * afastamento
        osso((p1[0] + nx, p1[1] + ny), (p2[0] + nx, p2[1] + ny), max(1, esp - 1))
        osso((p1[0] - nx, p1[1] - ny), (p2[0] - nx, p2[1] - ny), max(1, esp - 1))

    # --- Geometria do tronco ---
    qe_real, qd_real = ponto(pontos, IDX["E"]["quadril"]), ponto(pontos, IDX["D"]["quadril"])
    tronco = None
    if oe and od:
        quadris_visiveis = bool(qe_real and qd_real)
        if quadris_visiveis:
            qe, qd = qe_real, qd_real
        else:  # webcam costuma mostrar só da cintura para cima: estima a posição dos quadris
            qe, qd = (oe[0], oe[1] + 1.3 * escala), (od[0], od[1] + 1.3 * escala)
        topo, base = media(oe, od), media(qe, qd)
        tronco = dict(oe=oe, od=od, qe=qe, qd=qd, topo=topo, base=base,
                      comp=distancia(topo, base), quadris_visiveis=quadris_visiveis)

    # --- Crânio ---
    if "cranio" in partes:
        ore, ord_ = ponto(pontos, IDX_ORELHA["E"]), ponto(pontos, IDX_ORELHA["D"])
        if ore and ord_:
            largura = distancia(ore, ord_)
            angulo = math.degrees(math.atan2(ord_[1] - ore[1], ord_[0] - ore[0]))
            centro = media(ore, ord_)
            centro = (centro[0], centro[1] - 0.15 * largura)
            cv2.ellipse(camada, inteiro(centro), (int(largura * 0.62), int(largura * 0.82)),
                        angulo, 0, 360, COR_OSSO, esp, cv2.LINE_AA)
            for lado in ("E", "D"):
                olho = ponto(pontos, IDX_OLHO[lado])
                if olho:
                    cv2.circle(camada, inteiro(olho), max(3, int(largura * 0.12)),
                               COR_OSSO, max(1, esp - 1), cv2.LINE_AA)
            nariz = ponto(pontos, IDX_NARIZ)
            if nariz:
                cv2.circle(camada, inteiro(nariz), max(2, int(largura * 0.05)), COR_OSSO, -1, cv2.LINE_AA)
            be, bd = ponto(pontos, IDX_BOCA["E"]), ponto(pontos, IDX_BOCA["D"])
            if be and bd:  # mandíbula: orelha -> boca -> orelha
                mandibula = np.array([inteiro(ore), inteiro(be), inteiro(bd), inteiro(ord_)], np.int32)
                cv2.polylines(camada, [mandibula], False, COR_OSSO, esp, cv2.LINE_AA)

    # --- Coluna vertebral ---
    if tronco:
        topo, base = tronco["topo"], tronco["base"]

        def vertebras(p_ini, p_fim, qtd, t_ini, t_fim, tamanho):
            for t in np.linspace(t_ini, t_fim, qtd):
                c = lerp(p_ini, p_fim, float(t))
                cv2.ellipse(camada, inteiro(c), (int(esp * tamanho * 1.8), max(2, int(esp * tamanho))),
                            0, 0, 360, COR_OSSO, -1, cv2.LINE_AA)

        if "coluna_cervical" in partes:
            ore, ord_ = ponto(pontos, IDX_ORELHA["E"]), ponto(pontos, IDX_ORELHA["D"])
            cabeca = media(ore, ord_) if (ore and ord_) else (topo[0], topo[1] - 0.6 * escala)
            vertebras(cabeca, topo, 7, 0.1, 0.95, 0.8)
        if "coluna_dorsal" in partes:
            vertebras(topo, base, 12, 0.03, 0.62, 0.9)
        if "coluna_lombar" in partes:
            vertebras(topo, base, 5, 0.66, 0.98, 1.1)

        # --- Costelas, esterno e clavículas ---
        if "costelas" in partes:
            for k in range(8):
                t = 0.06 + k * 0.07
                centro = lerp(topo, base, t)
                fator = 0.62 + 0.28 * math.sin(math.pi * (k + 1) / 9)
                for ombro_p, quadril_p in ((tronco["oe"], tronco["qe"]), (tronco["od"], tronco["qd"])):
                    borda = lerp(centro, lerp(ombro_p, quadril_p, t), fator)
                    meio = lerp(centro, borda, 0.5)
                    pts = np.array([
                        inteiro(centro),
                        inteiro((meio[0], meio[1] - 0.015 * escala)),
                        inteiro((borda[0], borda[1] + 0.07 * escala)),
                    ], np.int32)
                    cv2.polylines(camada, [pts], False, COR_OSSO, max(1, esp - 1), cv2.LINE_AA)
        if "esterno" in partes:
            osso(lerp(topo, base, 0.03), lerp(topo, base, 0.33), esp + 1)

        for lado in dict(partes).get("clavicula", []):
            if lado:
                ombro_p = tronco["oe"] if lado == "E" else tronco["od"]
                osso(lerp(topo, base, 0.03), (ombro_p[0], ombro_p[1] - 0.03 * escala))

    # --- Pelve ---
    if "pelve" in partes and tronco and tronco["quadris_visiveis"]:
        qe, qd, base = tronco["qe"], tronco["qd"], tronco["base"]
        largura = distancia(qe, qd)
        for q in (qe, qd):
            centro = lerp(q, base, 0.12)
            cv2.ellipse(camada, inteiro((centro[0], centro[1] - 0.12 * largura)),
                        (int(largura * 0.30), int(largura * 0.22)), 0, 0, 360,
                        COR_OSSO, esp, cv2.LINE_AA)
            cv2.circle(camada, inteiro(q), max(3, int(largura * 0.09)), COR_OSSO, -1, cv2.LINE_AA)
        cv2.ellipse(camada, inteiro(base), (int(largura * 0.12), int(largura * 0.16)), 0, 0, 360,
                    COR_OSSO, esp, cv2.LINE_AA)

    # --- Membros ---
    for parte, lados in partes.items():
        for lado in lados:
            if lado is None or parte not in ("ombro", "braco", "antebraco", "mao", "coxa", "perna", "pe"):
                continue
            i = IDX[lado]
            p = lambda nome: ponto(pontos, i[nome])

            if parte == "ombro" and p("ombro"):
                cv2.circle(camada, inteiro(p("ombro")), max(4, int(escala * 0.08)), COR_OSSO, -1, cv2.LINE_AA)
            elif parte == "braco":
                osso(p("ombro"), p("cotovelo"), esp + 1)
            elif parte == "antebraco":
                osso_duplo(p("cotovelo"), p("punho"), 0.012 * escala)
            elif parte == "mao" and p("punho"):
                for dedo in ("polegar", "indicador", "mindinho"):
                    osso(p("punho"), p(dedo), max(1, esp - 1))
            elif parte == "coxa":
                osso(p("quadril"), p("joelho"), esp + 2)
                if p("joelho") and "perna" not in partes:
                    cv2.circle(camada, inteiro(p("joelho")), max(3, int(escala * 0.05)), COR_OSSO, -1, cv2.LINE_AA)
            elif parte == "perna":
                osso_duplo(p("joelho"), p("tornozelo"), 0.015 * escala)
            elif parte == "pe":
                osso(p("tornozelo"), p("calcanhar"), esp)
                osso(p("tornozelo"), p("ponta_pe"), esp)
                osso(p("calcanhar"), p("ponta_pe"), max(1, esp - 1))

    # Brilho suave ao redor dos ossos + ossos nítidos por cima
    brilho = cv2.GaussianBlur(camada, (0, 0), 6)
    resultado = cv2.addWeighted(imagem, 1.0, brilho, 0.7, 0)
    mascara = camada.any(axis=2)
    resultado[mascara] = camada[mascara]
    return resultado


# ----------------------------------------------------------------------
# Aplicação
# ----------------------------------------------------------------------
class SistemaRaioX:
    def __init__(self, root):
        self.root = root
        self.root.title("Raio-X Digital")
        self.root.geometry("950x720")
        self.root.configure(bg="#1e1e1e")

        self.frame_base_rx = None      # imagem de raio-x sem cabeçalho
        self.frame_atual_rx = None     # imagem de raio-x com cabeçalho
        self.congelado = False
        self.after_id = None
        self.incidencia_var = tk.StringVar(value="1")
        self.var_esqueleto = tk.BooleanVar(value=True)
        self.var_silhueta = tk.BooleanVar(value=True)
        self.status_pose = tk.StringVar(value="")

        self.landmarker = None
        self.ultimo_timestamp = 0
        self.iniciar_pose()

        self.cap = self.abrir_camera()

        self.criar_interface()

        if self.cap is not None:
            self.atualizar_video()
        else:
            self.label_video.configure(
                text="Câmera não encontrada.\nConecte uma câmera e reinicie o programa.",
                fg="white",
                font=("Arial", 14),
            )

    # ------------------------------------------------------------------
    # MediaPipe Pose Landmarker
    # ------------------------------------------------------------------
    def iniciar_pose(self):
        if not MEDIAPIPE_OK:
            self.status_pose.set("MediaPipe não instalado (pip install mediapipe)")
            return

        self.root.title("Raio-X Digital - preparando modelo de pose...")
        self.root.update()
        if not garantir_modelo():
            self.root.title("Raio-X Digital")
            self.status_pose.set("Modelo de pose indisponível (sem internet?)")
            messagebox.showwarning(
                "Modelo de pose",
                "Não foi possível baixar o modelo do Pose Landmarker.\n"
                f"Baixe manualmente em:\n{URL_MODELO}\n"
                f"e salve em:\n{CAMINHO_MODELO}\n\n"
                "O programa seguirá sem o esqueleto.",
            )
            return

        try:
            opcoes = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(CAMINHO_MODELO)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                output_segmentation_masks=True,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(opcoes)
            self.status_pose.set("Pose: aguardando imagem...")
        except Exception as erro:
            self.landmarker = None
            self.status_pose.set("Falha ao iniciar o Pose Landmarker")
            messagebox.showwarning("Pose Landmarker", f"Não foi possível iniciar o modelo:\n{erro}")
        self.root.title("Raio-X Digital")

    def detectar_pose(self, frame):
        """
        Detecta a pose no frame original (sem espelhar, para que esquerda/direita
        anatômicas fiquem corretas) e devolve (pontos, mascara) já espelhados
        para casar com a imagem exibida.
        """
        if self.landmarker is None:
            return None, None

        altura, largura = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # O modo VIDEO exige timestamps estritamente crescentes (em ms)
        timestamp = int(time.monotonic() * 1000)
        if timestamp <= self.ultimo_timestamp:
            timestamp = self.ultimo_timestamp + 1
        self.ultimo_timestamp = timestamp

        try:
            resultado = self.landmarker.detect_for_video(imagem_mp, timestamp)
        except Exception:
            return None, None

        if not resultado.pose_landmarks:
            return None, None

        pontos = [
            ((1.0 - lm.x) * largura, lm.y * altura, lm.visibility if lm.visibility is not None else 0.0)
            for lm in resultado.pose_landmarks[0]
        ]

        mascara = None
        if resultado.segmentation_masks:
            mascara = np.squeeze(resultado.segmentation_masks[0].numpy_view()).astype(np.float32)
            mascara = cv2.flip(mascara, 1)
            if mascara.shape[:2] != (altura, largura):
                mascara = cv2.resize(mascara, (largura, altura))
        return pontos, mascara

    @staticmethod
    def isolar_silhueta(imagem, mascara):
        """Escurece o fundo e mantém o efeito de raio-x só sobre a pessoa."""
        suave = np.clip(cv2.GaussianBlur(mascara, (0, 0), 3), 0.0, 1.0)[:, :, None]
        return (imagem * (0.12 + 0.88 * suave)).astype(np.uint8)

    # ------------------------------------------------------------------
    # Câmera
    # ------------------------------------------------------------------
    def abrir_camera(self):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # abre mais rápido no Windows
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            cap.release()
            messagebox.showerror(
                "Erro de câmera",
                "Não foi possível acessar a câmera.\n"
                "Verifique se ela está conectada e não está sendo usada por outro programa.",
            )
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def criar_interface(self):
        painel_esq = tk.Frame(self.root, bg="#2b2b2b", padx=20, pady=20, width=300)
        painel_esq.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            painel_esq, text="DADOS DO EXAME", font=("Arial", 14, "bold"),
            bg="#2b2b2b", fg="#00d4ff",
        ).pack(pady=(0, 15))

        tk.Label(painel_esq, text="Nome do Paciente:", bg="#2b2b2b", fg="white", anchor="w").pack(fill=tk.X)
        self.entry_nome = tk.Entry(painel_esq, font=("Arial", 11))
        self.entry_nome.pack(fill=tk.X, pady=(2, 10))

        tk.Label(painel_esq, text="Data de Nascimento (dd/mm/aaaa):", bg="#2b2b2b", fg="white", anchor="w").pack(fill=tk.X)
        self.entry_nasc = tk.Entry(painel_esq, font=("Arial", 11))
        self.entry_nasc.pack(fill=tk.X, pady=(2, 10))

        tk.Label(painel_esq, text="Local do RX:", bg="#2b2b2b", fg="white", anchor="w").pack(fill=tk.X)
        self.combo_local = ttk.Combobox(painel_esq, font=("Arial", 10), values=LISTA_EXAMES, width=32)
        self.combo_local.set("TORAX PA")
        self.combo_local.pack(fill=tk.X, pady=(2, 10))

        tk.Label(painel_esq, text="Incidências:", bg="#2b2b2b", fg="white", anchor="w").pack(fill=tk.X)
        frame_inc = tk.Frame(painel_esq, bg="#2b2b2b")
        frame_inc.pack(fill=tk.X, pady=(2, 10))

        for num in ["1", "2", "3"]:
            tk.Radiobutton(
                frame_inc,
                text=num,
                variable=self.incidencia_var,
                value=num,
                indicatoron=False,
                font=("Arial", 11, "bold"),
                bg="#444444",
                fg="white",
                selectcolor="#00a8cc",
                activebackground="#0088aa",
                activeforeground="white",
                bd=0,
                pady=6,
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        tk.Label(painel_esq, text="Data e Horário:", bg="#2b2b2b", fg="white", anchor="w").pack(fill=tk.X)
        self.entry_horario = tk.Entry(painel_esq, font=("Arial", 11))
        self.entry_horario.insert(0, datetime.now().strftime(FORMATO_DATA_HORA))
        self.entry_horario.pack(fill=tk.X, pady=(2, 10))

        tk.Button(
            painel_esq, text="Atualizar Horário", command=self.atualizar_hora,
            bg="#444", fg="white",
        ).pack(fill=tk.X, pady=(0, 10))

        # Opções do MediaPipe
        estado_pose = tk.NORMAL if self.landmarker is not None else tk.DISABLED
        for texto, variavel in (("Mostrar esqueleto (Pose)", self.var_esqueleto),
                                ("Isolar silhueta (fundo escuro)", self.var_silhueta)):
            tk.Checkbutton(
                painel_esq, text=texto, variable=variavel, state=estado_pose,
                bg="#2b2b2b", fg="white", selectcolor="#444444",
                activebackground="#2b2b2b", activeforeground="white",
                disabledforeground="#777777", anchor="w",
            ).pack(fill=tk.X)

        tk.Label(
            painel_esq, textvariable=self.status_pose, bg="#2b2b2b", fg="#aaaaaa",
            font=("Arial", 9), wraplength=260, justify="left", anchor="w",
        ).pack(fill=tk.X, pady=(4, 10))

        self.btn_capturar = tk.Button(
            painel_esq, text="CAPTURAR E GERAR RX", font=("Arial", 12, "bold"),
            bg="#00a8cc", fg="white", height=2, command=self.salvar_exame,
        )
        self.btn_capturar.pack(fill=tk.X, pady=5)

        painel_dir = tk.Frame(self.root, bg="#121212")
        painel_dir.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.label_video = tk.Label(painel_dir, bg="black")
        self.label_video.pack(expand=True)

    def atualizar_hora(self):
        self.entry_horario.delete(0, tk.END)
        self.entry_horario.insert(0, datetime.now().strftime(FORMATO_DATA_HORA))

    # ------------------------------------------------------------------
    # Processamento de imagem
    # ------------------------------------------------------------------
    def processar_raiox(self, frame):
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bordas = cv2.Laplacian(cinza, cv2.CV_8U, ksize=3)
        negativo = cv2.bitwise_not(cinza)
        combinado = cv2.addWeighted(negativo, 0.85, bordas, 0.6, 0)
        contraste = cv2.convertScaleAbs(combinado, alpha=1.35, beta=-15)
        return cv2.applyColorMap(contraste, cv2.COLORMAP_BONE)

    def adicionar_cabecalho_medico(self, imagem):
        img_final = imagem.copy()
        altura, largura = img_final.shape[:2]

        nome = remover_acentos(self.entry_nome.get()).upper() or "PACIENTE NAO IDENTIFICADO"
        nasc = self.entry_nasc.get().strip() or "--/--/----"
        local = remover_acentos(self.combo_local.get()).upper()
        incidencia = self.incidencia_var.get()
        horario = self.entry_horario.get()

        fonte = cv2.FONT_HERSHEY_SIMPLEX
        cor = (255, 255, 255)

        # Faixa escura semitransparente atrás do cabeçalho (melhora a leitura)
        overlay = img_final.copy()
        cv2.rectangle(overlay, (0, 0), (largura, 140), (0, 0, 0), -1)
        img_final = cv2.addWeighted(overlay, 0.55, img_final, 0.45, 0)

        linhas = [
            f"PACIENTE: {nome}",
            f"NASC: {nasc}",
            f"EXAME: RX {local}",
            f"INCIDENCIA: {incidencia}",
            f"DATA/HORA: {horario}",
        ]
        for i, texto in enumerate(linhas):
            cv2.putText(img_final, texto, (15, 25 + i * 25), fonte, 0.50, cor, 1, cv2.LINE_AA)

        # Marca de simulação no rodapé
        overlay = img_final.copy()
        cv2.rectangle(overlay, (0, altura - 28), (largura, altura), (0, 0, 0), -1)
        img_final = cv2.addWeighted(overlay, 0.55, img_final, 0.45, 0)
        cv2.putText(
            img_final, "SIMULACAO - NAO E UM EXAME MEDICO",
            (15, altura - 9), fonte, 0.50, (0, 200, 255), 1, cv2.LINE_AA,
        )

        return img_final

    # ------------------------------------------------------------------
    # Vídeo
    # ------------------------------------------------------------------
    def mostrar_imagem(self, imagem_bgr):
        img_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        self.label_video.imgtk = img_tk  # mantém a referência para não ser coletada
        self.label_video.configure(image=img_tk)

    def atualizar_video(self):
        if not self.congelado:
            ret, frame = self.cap.read()
            if ret:
                usar_pose = self.landmarker is not None and (
                    self.var_esqueleto.get() or self.var_silhueta.get())

                pontos, mascara = self.detectar_pose(frame) if usar_pose else (None, None)
                frame = cv2.flip(frame, 1)

                base = self.processar_raiox(frame)

                if usar_pose:
                    if pontos is None:
                        self.status_pose.set("Nenhuma pessoa detectada. Fique de frente para a câmera.")
                    else:
                        self.status_pose.set("Pose detectada.")
                        if self.var_silhueta.get() and mascara is not None:
                            base = self.isolar_silhueta(base, mascara)
                        if self.var_esqueleto.get():
                            base = desenhar_esqueleto(base, pontos, partes_do_exame(self.combo_local.get()))

                self.frame_base_rx = base
                self.frame_atual_rx = self.adicionar_cabecalho_medico(self.frame_base_rx)
                self.mostrar_imagem(self.frame_atual_rx)

        self.after_id = self.root.after(INTERVALO_VIDEO_MS, self.atualizar_video)

    def descongelar(self):
        self.congelado = False
        self.btn_capturar.configure(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Salvar exame
    # ------------------------------------------------------------------
    def salvar_exame(self):
        if self.frame_base_rx is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem disponível para salvar.")
            return

        if not data_nascimento_valida(self.entry_nasc.get()):
            messagebox.showwarning(
                "Data inválida",
                "Informe a data de nascimento no formato dd/mm/aaaa (e que não seja futura).",
            )
            return

        # Atualiza o horário no momento exato da captura
        self.atualizar_hora()
        agora = datetime.now()

        # Recompõe o cabeçalho com o horário atualizado
        imagem_final = self.adicionar_cabecalho_medico(self.frame_base_rx)

        nome = limpar_nome(self.entry_nome.get())
        incidencia = self.incidencia_var.get()

        pasta = PASTA_BASE / nome / agora.strftime("%Y-%m-%d")
        try:
            pasta.mkdir(parents=True, exist_ok=True)  # cria a pasta se não existir
        except OSError as erro:
            messagebox.showerror("Erro", f"Não foi possível criar a pasta:\n{pasta}\n\n{erro}")
            return

        arquivo = pasta / f"RX_{nome}_INC{incidencia}_{agora:%H%M%S}.png"

        # imwrite não gera exceção: retorna False quando falha
        if not cv2.imwrite(str(arquivo), imagem_final):
            messagebox.showerror("Erro", f"Não foi possível salvar a imagem:\n{arquivo}")
            return

        # Congela a imagem por um instante como feedback visual da captura
        self.frame_atual_rx = imagem_final
        self.mostrar_imagem(imagem_final)
        self.congelado = True
        self.btn_capturar.configure(state=tk.DISABLED)
        self.root.after(TEMPO_CONGELADO_MS, self.descongelar)

        messagebox.showinfo("Exame Salvo", f"Imagem salva em:\n{arquivo}")

    # ------------------------------------------------------------------
    # Encerramento
    # ------------------------------------------------------------------
    def fechar(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        if self.cap is not None:
            self.cap.release()
        if self.landmarker is not None:
            self.landmarker.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaRaioX(root)
    root.protocol("WM_DELETE_WINDOW", app.fechar)
    root.mainloop()

"""
Motor de radiografia SIMULADA.

A partir de uma foto (pose do corpo, mãos) monta um modelo anatômico procedural
(ossos, pulmões, seios da face, etc.), calcula a atenuação dos raios-X como uma
soma de densidades (tecido mole + osso + ar) e converte isso numa imagem com
aparência de radiografia: janela de contraste, ruído, colimação e marcador D/E.

Não é um exame médico: é uma ilustração gerada por regras anatômicas.
"""
import math
import unicodedata

import cv2
import numpy as np

# ----------------------------------------------------------------------
# Pontos do corpo
# ----------------------------------------------------------------------
IDX_POSE = {
    "nariz": 0, "olho_E": 2, "olho_D": 5, "orelha_E": 7, "orelha_D": 8,
    "boca_E": 9, "boca_D": 10,
    "ombro_E": 11, "ombro_D": 12, "cotovelo_E": 13, "cotovelo_D": 14,
    "punho_E": 15, "punho_D": 16,
    "quadril_E": 23, "quadril_D": 24, "joelho_E": 25, "joelho_D": 26,
    "tornozelo_E": 27, "tornozelo_D": 28,
    "calcanhar_E": 29, "calcanhar_D": 30, "ponta_pe_E": 31, "ponta_pe_D": 32,
}

# Posição padrão (unidades de largura de ombros; origem no meio dos ombros; y para baixo)
CANON = {
    "ombro_E": (0.5, 0.0), "ombro_D": (-0.5, 0.0),
    "cotovelo_E": (0.62, 0.75), "cotovelo_D": (-0.62, 0.75),
    "punho_E": (0.72, 1.38), "punho_D": (-0.72, 1.38),
    "quadril_E": (0.22, 1.25), "quadril_D": (-0.22, 1.25),
    "joelho_E": (0.25, 2.35), "joelho_D": (-0.25, 2.35),
    "tornozelo_E": (0.27, 3.35), "tornozelo_D": (-0.27, 3.35),
    "calcanhar_E": (0.28, 3.45), "calcanhar_D": (-0.28, 3.45),
    "ponta_pe_E": (0.30, 4.05), "ponta_pe_D": (-0.30, 4.05),
    "nariz": (0.0, -0.42), "olho_E": (0.09, -0.50), "olho_D": (-0.09, -0.50),
    "orelha_E": (0.20, -0.46), "orelha_D": (-0.20, -0.46),
    "boca_E": (0.05, -0.34), "boca_D": (-0.05, -0.34),
}
ORDEM = [
    ("ombro_E", None), ("ombro_D", None),
    ("cotovelo_E", "ombro_E"), ("cotovelo_D", "ombro_D"),
    ("punho_E", "cotovelo_E"), ("punho_D", "cotovelo_D"),
    ("quadril_E", "ombro_E"), ("quadril_D", "ombro_D"),
    ("joelho_E", "quadril_E"), ("joelho_D", "quadril_D"),
    ("tornozelo_E", "joelho_E"), ("tornozelo_D", "joelho_D"),
    ("calcanhar_E", "tornozelo_E"), ("calcanhar_D", "tornozelo_D"),
    ("ponta_pe_E", "calcanhar_E"), ("ponta_pe_D", "calcanhar_D"),
    ("nariz", None), ("olho_E", None), ("olho_D", None),
    ("orelha_E", None), ("orelha_D", None), ("boca_E", None), ("boca_D", None),
]

# Mão padrão (mão direita, vista dorsal, dedos para cima; unidades = comprimento da mão)
MAO_MOLDE = np.array([
    (0.00, 0.00),
    (-0.17, 0.12), (-0.30, 0.30), (-0.40, 0.47), (-0.47, 0.62),
    (-0.17, 0.50), (-0.20, 0.70), (-0.21, 0.82), (-0.22, 0.93),
    (-0.03, 0.52), (-0.03, 0.74), (-0.03, 0.87), (-0.03, 1.00),
    (0.11, 0.50), (0.13, 0.70), (0.14, 0.82), (0.15, 0.92),
    (0.24, 0.44), (0.28, 0.60), (0.30, 0.70), (0.31, 0.79),
])


def remover_acentos(t):
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 1.0])


def suave(pts, it=2, fechado=True):
    """Suaviza um polígono (Chaikin) para formas orgânicas."""
    P = [np.asarray(p, float) for p in pts]
    for _ in range(it):
        N, n = [], len(P)
        for i in range(n if fechado else n - 1):
            a, b = P[i], P[(i + 1) % n]
            N += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        P = N if fechado else [P[0]] + N + [P[-1]]
    return [tuple(p) for p in P]


class Corpo:
    """Pontos anatômicos da pessoa (detectados ou preenchidos com proporções padrão)."""

    def __init__(self, largura, altura, pose=None, maos=None):
        self.W, self.H = largura, altura
        P, reais = {}, set()
        if pose:
            for nome, i in IDX_POSE.items():
                x, y, v = pose[i]
                if v >= 0.5 and -0.05 * largura <= x <= 1.05 * largura and -0.05 * altura <= y <= 1.05 * altura:
                    P[nome] = np.array([x, y], float)
                    reais.add(nome)

        oe, od = P.get("ombro_E"), P.get("ombro_D")
        if oe is not None and od is not None and np.linalg.norm(oe - od) > 30:
            M = (oe + od) / 2
            ex = oe - od
        else:
            for k in ("ombro_E", "ombro_D"):
                P.pop(k, None)
                reais.discard(k)
            ex = np.array([largura * 0.30, 0.0])
            M = np.array([largura * 0.5, altura * 0.30])
        sw = float(np.linalg.norm(ex))
        exu = ex / sw
        eyu = np.array([-exu[1], exu[0]])

        for nome, pai in ORDEM:
            if nome in P:
                continue
            cx, cy = CANON[nome]
            if pai:
                px, py = CANON[pai]
                P[nome] = P[pai] + sw * (exu * (cx - px) + eyu * (cy - py))
            else:
                P[nome] = M + sw * (exu * cx + eyu * cy)

        # eixo do tronco: usa os quadris reais quando existirem
        ey = eyu * sw
        if "quadril_E" in reais and "quadril_D" in reais:
            cand = ((P["quadril_E"] + P["quadril_D"]) / 2 - M) / 1.25
            if 0.6 * sw < np.linalg.norm(cand) < 1.6 * sw:
                ey = cand
        self.M, self.sw, self.ex, self.ey = M, sw, ex, ey
        self.reais = reais
        self._ex_front, self._ey_front = ex.copy(), ey.copy()
        self.lateral = False
        P["quadril_E"] = self.tq(0.22, 1.25)
        P["quadril_D"] = self.tq(-0.22, 1.25)
        self.P = P

        # cabeça (unidade L = distância olhos-boca)
        olhos = (P["olho_E"] + P["olho_D"]) / 2
        boca = (P["boca_E"] + P["boca_D"]) / 2
        L = float(np.linalg.norm(olhos - boca))
        if "olho_E" not in reais or "boca_E" not in reais or L < 0.03 * sw:
            olhos = M + sw * (exu * 0 + eyu * -0.50)
            boca = M + sw * (eyu * -0.34)
            L = 0.16 * sw
        self.L = L
        self.cab0 = olhos
        self.cab_y = unit(boca - olhos)
        self.cab_x = np.array([self.cab_y[1], -self.cab_y[0]])

        # mãos
        self.mao = {}
        self.mao_real = {}
        maos = maos or {}
        for lado in ("E", "D"):
            if lado in maos:
                H = np.asarray(maos[lado], float)
                self.mao_real[lado] = True
                comp = float(np.linalg.norm(H[12] - H[0]))
                self.mao[lado] = H
                P["punho_" + lado] = H[0].copy()
                u = unit(H[9] - H[0])
                P["cotovelo_" + lado] = H[0] - u * (comp / 0.48) * 0.63
                self.sw_mao = getattr(self, "sw_mao", {})
                self.sw_mao[lado] = comp / 0.48
            else:
                self.mao_real[lado] = False
                self.mao[lado] = self._molde_mao(lado)
        self.sw_mao = getattr(self, "sw_mao", {})
        for lado in ("E", "D"):
            self.sw_mao.setdefault(lado, sw)

    def _molde_mao(self, lado):
        u = unit(self.P["punho_" + lado] - self.P["cotovelo_" + lado])
        perp = np.array([-u[1], u[0]])
        Hc = 0.48 * self.sw
        pts = MAO_MOLDE.copy()
        if lado == "E":
            pts[:, 0] *= -1
        return np.array([self.P["punho_" + lado] + Hc * (x * perp + y * u) for x, y in pts])

    # coordenadas do tronco: a = lateral (unidades de ombro), b = descendo a partir dos ombros
    def tq(self, a, b):
        return self.M + a * self.ex + b * self.ey

    @property
    def ang_ex(self):
        return math.degrees(math.atan2(self.ex[1], self.ex[0]))

    def modo_lateral(self):
        """Perfil: o eixo horizontal passa a ser a profundidade (face para a esquerda)."""
        if self.lateral:
            return
        eyu = unit(self.ey)
        self.ex = np.array([eyu[1], -eyu[0]]) * self.sw
        self.lateral = True

    def cab(self, x, y):
        """Coordenadas da cabeça (x lateral, y para baixo, em unidades de L)."""
        return self.cab0 + self.L * (x * self.cab_x + y * self.cab_y)


# ----------------------------------------------------------------------
# Renderizador
# ----------------------------------------------------------------------
class Render:
    def __init__(self, eixo, pts=None, centro=None, meia=None, saida=1100,
                 pad=0.10, prop=(0.5, 2.0), margem=0.07, semente=1):
        base, topo = np.asarray(eixo[0], float), np.asarray(eixo[1], float)
        v = topo - base
        th = -math.pi / 2 - math.atan2(v[1], v[0])
        self.th_deg = math.degrees(th)
        self._c, self._s = math.cos(th), math.sin(th)
        if centro is not None:
            q = self.rot(centro)
            x0, x1, y0, y1 = q[0] - meia[0], q[0] + meia[0], q[1] - meia[1], q[1] + meia[1]
        else:
            Q = np.array([self.rot(p) for p in pts])
            x0, y0 = Q.min(0)
            x1, y1 = Q.max(0)
        p = max(x1 - x0, y1 - y0) * pad
        x0, x1, y0, y1 = x0 - p, x1 + p, y0 - p, y1 + p
        w, h = x1 - x0, y1 - y0
        lo, hi = prop
        if w / h < lo:
            e = (lo * h - w) / 2
            x0, x1 = x0 - e, x1 + e
        elif w / h > hi:
            e = (w / hi - h) / 2
            y0, y1 = y0 - e, y1 + e
        fw, fh = x1 - x0, y1 - y0
        cw, ch = fw * (1 + 2 * margem), fh * (1 + 2 * margem)
        self.S = saida / max(cw, ch)
        self.ox, self.oy = x0 - fw * margem, y0 - fh * margem
        self.W, self.H = int(round(cw * self.S)), int(round(ch * self.S))
        self.campo = ((x0 - self.ox) * self.S, (y0 - self.oy) * self.S,
                      (x1 - self.ox) * self.S, (y1 - self.oy) * self.S)
        z = lambda: np.zeros((self.H, self.W), np.float32)
        self.osso, self.cortex, self.ar, self.sombra, self.tec = z(), z(), z(), z(), z()
        self.rng = np.random.default_rng(semente)
        self.marca = None

    # --- geometria ---
    def rot(self, p):
        x, y = float(p[0]), float(p[1])
        return (self._c * x - self._s * y, self._s * x + self._c * y)

    def Pt(self, p):
        q = self.rot(p)
        return (int(round((q[0] - self.ox) * self.S)), int(round((q[1] - self.oy) * self.S)))

    def P(self, pts):
        return np.array([self.Pt(p) for p in pts], np.int32)

    def _t(self):
        return np.zeros((self.H, self.W), np.uint8)

    @staticmethod
    def _add(camada, t, peso):
        camada += t.astype(np.float32) * (peso / 255.0)

    # --- primitivas ---
    def poligono(self, camada, pts, peso):
        t = self._t()
        cv2.fillPoly(t, [self.P(pts)], 255, cv2.LINE_AA)
        self._add(camada, t, peso)

    def contorno(self, camada, pts, esp_px, peso, fechado=True):
        t = self._t()
        cv2.polylines(t, [self.P(pts)], fechado, 255, max(1, int(round(esp_px))), cv2.LINE_AA)
        self._add(camada, t, peso)

    def linha(self, camada, a, b, esp, peso):
        t = self._t()
        cv2.line(t, self.Pt(a), self.Pt(b), 255, max(1, int(round(esp * self.S))), cv2.LINE_AA)
        self._add(camada, t, peso)

    def polilinha(self, camada, pts, esp, peso):
        t = self._t()
        cv2.polylines(t, [self.P(pts)], False, 255, max(1, int(round(esp * self.S))), cv2.LINE_AA)
        self._add(camada, t, peso)

    def elipse(self, camada, c, ex, ey, ang, peso, esp_px=None):
        t = self._t()
        eixos = (max(1, int(round(ex * self.S))), max(1, int(round(ey * self.S))))
        cv2.ellipse(t, self.Pt(c), eixos, ang + self.th_deg, 0, 360, 255,
                    -1 if esp_px is None else max(1, int(round(esp_px))), cv2.LINE_AA)
        self._add(camada, t, peso)

    def osso_poly(self, pts, fill=0.5, cort=0.6, esp=2.4):
        self.poligono(self.osso, pts, fill)
        if cort:
            self.contorno(self.cortex, pts, esp, cort)

    def osso_longo(self, a, b, w_dia, w_a, w_b, gap=0.03, fill=0.5, cort=0.65, canal=True):
        """Osso longo: diáfise com canal medular e epífises alargadas."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        if L < 2:
            return
        u = (b - a) / L
        n = np.array([-u[1], u[0]])
        a2, L2 = a + u * L * gap, L * (1 - 2 * gap)
        ts = np.linspace(0, 1, 30)

        def perfil(t, f=1.0):
            return f * (w_dia + (w_a - w_dia) * math.exp(-t / 0.10) + (w_b - w_dia) * math.exp(-(1 - t) / 0.10))

        esq = [a2 + u * L2 * t + n * perfil(t) / 2 for t in ts]
        dir_ = [a2 + u * L2 * t - n * perfil(t) / 2 for t in ts]
        self.osso_poly(esq + dir_[::-1], fill, cort)
        if canal:
            tt = [t for t in ts if 0.22 <= t <= 0.78]
            i_esq = [a2 + u * L2 * t + n * w_dia * 0.20 for t in tt]
            i_dir = [a2 + u * L2 * t - n * w_dia * 0.20 for t in tt]
            self.poligono(self.osso, i_esq + i_dir[::-1], -fill * 0.45)

    # --- composição final ---
    def finalizar(self, marca=None):
        H, W = self.H, self.W
        ruido = self.rng.normal(size=(H, W)).astype(np.float32)
        trab = cv2.GaussianBlur(ruido, (0, 0), 2.2)
        trab /= trab.std() + 1e-6
        ar = np.clip(cv2.GaussianBlur(self.ar, (0, 0), 1.6), 0, 1)
        tec = self.tec * (1 - 0.85 * ar)
        osso = np.clip(self.osso, 0, None) * (1 + 0.20 * trab) + self.cortex
        total = tec + 0.85 * osso + 0.45 * cv2.GaussianBlur(self.sombra, (0, 0), 2.0)
        total = cv2.GaussianBlur(total, (0, 0), 1.2)

        fx0, fy0, fx1, fy1 = [int(round(v)) for v in self.campo]
        campo = np.zeros((H, W), np.float32)
        campo[max(fy0, 0):fy1, max(fx0, 0):fx1] = 1
        v = total[campo > 0]
        lo, hi = np.percentile(v, 1.5), np.percentile(v, 99.7)
        B = np.clip((total - lo) / (hi - lo + 1e-6), 0, 1) ** 0.85
        B = 1 / (1 + np.exp(-(B - 0.5) * 4.2))
        B = (B - B.min()) / (B.max() - B.min() + 1e-6)
        B = 0.92 * B + 0.08 * cv2.GaussianBlur(B, (0, 0), 30)

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        B *= 1 - 0.16 * np.clip(r2, 0, 1.5)

        if marca:
            esc = max(1.2, W / 650)
            tam = cv2.getTextSize(marca[0], cv2.FONT_HERSHEY_DUPLEX, esc, 3)[0]
            x = fx0 + 18 if marca[1] == "esq" else fx1 - 18 - tam[0]
            m = np.zeros((H, W), np.float32)
            cv2.putText(m, marca[0], (int(x), int(fy0 + 22 + tam[1])), cv2.FONT_HERSHEY_DUPLEX, esc, 1.0, 3, cv2.LINE_AA)
            B = np.maximum(B, cv2.GaussianBlur(m, (0, 0), 0.8) * 0.95)

        B += self.rng.normal(0, 0.014, size=(H, W)).astype(np.float32)
        cs = cv2.GaussianBlur(campo, (0, 0), 3)
        B = B * cs + 0.035 * (1 - cs)
        return (np.clip(B, 0, 1) * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Construtores anatômicos
# ----------------------------------------------------------------------
def silhueta(R, c, lateral=False):
    """Tecido mole (contorno do corpo) com espessura proporcional à distância da borda."""
    sw, T = c.sw, c.tq
    m = R._t()

    def poli(pts):
        cv2.fillPoly(m, [R.P(pts)], 255, cv2.LINE_AA)

    def cap(a, b, ra, rb):
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = unit(b - a)
        n = np.array([-u[1], u[0]])
        poli([a + n * ra, b + n * rb, b - n * rb, a - n * ra])
        for p, r in ((a, ra), (b, rb)):
            cv2.circle(m, R.Pt(p), max(1, int(r * R.S)), 255, -1, cv2.LINE_AA)

    if lateral:
        tronco = [(-0.30, -0.20), (0.32, -0.20), (0.36, 0.5), (0.30, 1.0), (0.36, 1.5), (-0.28, 1.5),
                  (-0.30, 0.9), (-0.34, 0.4)]
        poli([T(*p) for p in suave(tronco, 2)])
        cap(T(0.0, -0.1), c.cab(0, 1.0), 0.16 * sw, 0.16 * sw)
        poli([c.cab(*p) for p in suave([(-1.55, -0.2), (-1.3, -1.5), (-0.4, -2.1), (0.9, -2.0), (1.55, -1.0),
                                       (1.6, 0.2), (1.1, 1.0), (0.3, 1.6), (-0.8, 1.6), (-1.5, 1.0), (-1.9, 0.5)], 2)])
    else:
        tronco = [(-0.17, -0.55), (0.17, -0.55), (0.20, -0.25), (0.40, -0.10), (0.58, 0.04), (0.55, 0.30),
                  (0.49, 0.60), (0.43, 0.95), (0.50, 1.25), (0.56, 1.55), (0.2, 1.72), (-0.2, 1.72),
                  (-0.56, 1.55), (-0.50, 1.25), (-0.43, 0.95), (-0.49, 0.60), (-0.55, 0.30), (-0.58, 0.04),
                  (-0.40, -0.10), (-0.20, -0.25)]
        poli([T(*p) for p in suave(tronco, 2)])
        poli([c.cab(*p) for p in suave([(0, -2.0), (0.95, -1.55), (1.25, -0.5), (1.05, 0.6), (0.75, 1.4),
                                       (0.0, 1.9), (-0.75, 1.4), (-1.05, 0.6), (-1.25, -0.5), (-0.95, -1.55)], 2)])
        cap(T(0.0, -0.25), c.cab(0, 1.2), 0.15 * sw, 0.15 * sw)
        for l in ("E", "D"):
            P = c.P
            cap(P["ombro_" + l], P["cotovelo_" + l], 0.12 * sw, 0.09 * sw)
            cap(P["cotovelo_" + l], P["punho_" + l], 0.09 * sw, 0.06 * sw)
            cap(P["quadril_" + l], P["joelho_" + l], 0.20 * sw, 0.12 * sw)
            cap(P["joelho_" + l], P["tornozelo_" + l], 0.12 * sw, 0.07 * sw)
            cap(P["calcanhar_" + l], P["ponta_pe_" + l], 0.08 * sw, 0.07 * sw)
            H = c.mao[l]
            rr = 0.030 * c.sw_mao[l]
            poli([H[0], H[1], H[5], H[9], H[13], H[17]])
            for ini in (1, 5, 9, 13, 17):
                for k in range(3):
                    cap(H[ini + k], H[ini + k + 1], rr * 1.1, rr * 0.9)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5) / (R.S * sw)
    tec = 2.6 * np.minimum(d, 0.30) ** 0.8
    R.tec = cv2.GaussianBlur(tec.astype(np.float32), (0, 0), 2.0)


# ---------------- coluna ----------------
def vertebra_frontal(R, c, b, w, h, tipo):
    sw, T, ang = c.sw, c.tq, c.ang_ex
    hh = h * 0.5 * 0.84
    pts = suave([(-w / 2, b - hh), (w / 2, b - hh), (w * 0.44, b), (w / 2, b + hh), (-w / 2, b + hh), (-w * 0.44, b)], 2)
    R.osso_poly([T(*p) for p in pts], 0.50, 0.55)
    for s in (-1, 1):
        pc = T(s * w * 0.30, b - hh * 0.35)
        R.elipse(R.osso, pc, w * sw * 0.09, h * sw * 0.13, ang, 0.45)
        R.elipse(R.osso, pc, w * sw * 0.045, h * sw * 0.06, ang, -0.25)
        if tipo != "cervical":
            R.linha(R.osso, T(s * w * 0.55, b - hh * 0.9), T(s * w * 1.0, b - hh * 0.9), h * sw * 0.16, 0.30)
    R.elipse(R.osso, T(0, b + hh * 0.1), w * sw * 0.055, h * sw * 0.30, ang, 0.35)


def coluna_frontal(R, c, trechos=("cervical", "toracica", "lombar", "sacro")):
    if "cervical" in trechos:
        for i in range(7):
            vertebra_frontal(R, c, -0.46 + 0.055 * i, 0.11 if i == 0 else 0.055 + 0.003 * i, 0.04, "cervical")
    if "toracica" in trechos:
        for i in range(12):
            vertebra_frontal(R, c, -0.07 + 0.059 * i, 0.062 + 0.004 * i, 0.047, "toracica")
    if "lombar" in trechos:
        for i in range(5):
            vertebra_frontal(R, c, 0.66 + 0.09 * i, 0.115 + 0.012 * i, 0.075, "lombar")
    if "sacro" in trechos:
        T = c.tq
        pts = suave([(-0.12, 1.02), (0.12, 1.02), (0.135, 1.10), (0.09, 1.22), (0.04, 1.33),
                     (-0.04, 1.33), (-0.09, 1.22), (-0.135, 1.10)], 2)
        R.osso_poly([T(*p) for p in pts], 0.45, 0.55)
        for s in (-1, 1):
            for yy, rr in ((1.10, 0.014), (1.18, 0.012), (1.25, 0.010)):
                R.elipse(R.osso, T(s * 0.05, yy), rr * c.sw, rr * c.sw, 0, -0.22)
        R.osso_poly([T(*p) for p in [(-0.03, 1.34), (0.03, 1.34), (0.02, 1.42), (-0.02, 1.42)]], 0.3, 0.4)


def curva_lateral(b, flex=0.0):
    """Posição horizontal (profundidade) da coluna em perfil: lordose, cifose, lordose."""
    if b < -0.08:
        a = -0.035 * math.sin(math.pi * np.clip((b + 0.50) / 0.42, 0, 1))
    elif b < 0.66:
        a = 0.050 * math.sin(math.pi * (b + 0.08) / 0.74)
    else:
        a = -0.045 * math.sin(math.pi * np.clip((b - 0.60) / 0.60, 0, 1))
    if b < 0:
        a += flex * 0.35 * (-b) ** 1.2
    return a


def vertebra_lateral(R, c, b, a0, prof, alt, tipo, flex=0.0):
    sw, T = c.sw, c.tq
    pc = T(a0 + curva_lateral(b, flex), b)
    t = unit(T(a0 + curva_lateral(b + 0.01, flex), b + 0.01) - T(a0 + curva_lateral(b - 0.01, flex), b - 0.01))
    n = np.array([t[1], -t[0]])
    hh, d = alt * 0.5 * 0.84 * sw, prof * 0.5 * sw
    pts = suave([pc - n * d * 1.05 - t * hh, pc + n * d - t * hh, pc + n * d * 0.9 + t * hh,
                 pc - n * d * 1.05 + t * hh, pc - n * d * 0.85], 2)
    R.osso_poly(pts, 0.52, 0.6)
    ped = pc + n * (d + 0.015 * sw)
    R.elipse(R.osso, ped, 0.035 * sw, alt * 0.35 * sw, math.degrees(math.atan2(t[1], t[0])), 0.35)
    if tipo == "cervical":
        ponta = pc + n * (d + 0.07 * sw) + t * 0.01 * sw
    elif tipo == "toracica":
        ponta = pc + n * (d + 0.10 * sw) + t * 0.07 * sw
    else:
        ponta = pc + n * (d + 0.13 * sw)
    R.polilinha(R.osso, [pc + n * (d + 0.01 * sw), ponta], alt * 0.22 * sw, 0.40)


def coluna_lateral(R, c, trechos=("cervical", "toracica", "lombar", "sacro"), a0=0.0, flex=0.0):
    if "cervical" in trechos:
        for i in range(7):
            vertebra_lateral(R, c, -0.46 + 0.055 * i, a0, 0.07 + 0.004 * i, 0.04, "cervical", flex)
    if "toracica" in trechos:
        for i in range(12):
            vertebra_lateral(R, c, -0.07 + 0.059 * i, a0, 0.075 + 0.005 * i, 0.047, "toracica", flex)
    if "lombar" in trechos:
        for i in range(5):
            vertebra_lateral(R, c, 0.66 + 0.09 * i, a0, 0.13 + 0.01 * i, 0.075, "lombar", flex)
    if "sacro" in trechos:
        T = c.tq
        pts = suave([(a0 - 0.07, 1.04), (a0 + 0.06, 1.04), (a0 + 0.12, 1.20), (a0 + 0.10, 1.33),
                     (a0 + 0.05, 1.36), (a0 - 0.03, 1.28), (a0 - 0.08, 1.12)], 2)
        R.osso_poly([T(*p) for p in pts], 0.45, 0.55)


# ---------------- tórax ----------------
def costelas(R, c, n=10, peso=1.0):
    sw, T = c.sw, c.tq
    f = [0.55, 0.70, 0.82, 0.90, 0.96, 1.0, 1.0, 0.97, 0.92, 0.85, 0.70, 0.55]
    for k in range(n):
        y0 = -0.05 + 0.058 * k
        for s in (-1, 1):
            P0 = np.array([s * 0.04, y0])
            P1 = np.array([s * 0.68 * f[k], y0 + 0.06])
            P2 = np.array([s * 0.10, y0 + 0.31 + 0.01 * k])
            ts = np.linspace(0, 0.86, 26)
            cur = [(1 - t) ** 2 * P0 + 2 * t * (1 - t) * P1 + t * t * P2 for t in ts]
            for ini, fim, pf in ((0, 15, 1.0), (14, 26, 0.55)):
                seg = cur[ini:fim]
                if len(seg) < 2:
                    continue
                w = 0.026 * (1 - 0.3 * ini / 26)
                pts_c = [T(*p) for p in seg]
                nrm = []
                for i in range(len(pts_c)):
                    a_ = pts_c[max(i - 1, 0)]
                    b_ = pts_c[min(i + 1, len(pts_c) - 1)]
                    u = unit(b_ - a_)
                    nrm.append(np.array([-u[1], u[0]]))
                esq = [p + nn * w * sw / 2 for p, nn in zip(pts_c, nrm)]
                dd = [p - nn * w * sw / 2 for p, nn in zip(pts_c, nrm)]
                R.osso_poly(esq + dd[::-1], 0.30 * pf * peso, 0.42 * pf * peso, 2.0)


def marcas_vasculares(R, c, lado, hilo):
    sw, T = c.sw, c.tq
    s = 1 if lado == "E" else -1
    for _ in range(16):
        ang = R.rng.uniform(-1.2, 1.7)
        p = np.array(hilo, float)
        passo = R.rng.uniform(0.05, 0.075)
        larg = 0.016
        for _k in range(4):
            ang += R.rng.normal(0, 0.18)
            q = p + np.array([s * math.cos(ang), math.sin(ang)]) * passo
            if abs(q[0]) > 0.37:
                break
            R.linha(R.sombra, T(*p), T(*q), larg * sw, 0.10)
            p, larg = q, larg * 0.75


def torax_frontal(R, c, enfase=None):
    sw, T = c.sw, c.tq
    ang = c.ang_ex
    # pulmões (ar)
    pD = [(-0.14, -0.16), (-0.08, -0.08), (-0.065, 0.10), (-0.07, 0.30), (-0.08, 0.40), (-0.16, 0.36), (-0.26, 0.38),
          (-0.34, 0.47), (-0.39, 0.55), (-0.41, 0.30), (-0.39, 0.02), (-0.31, -0.10), (-0.22, -0.16)]
    pE = [(0.14, -0.16), (0.08, -0.08), (0.07, 0.12), (0.12, 0.28), (0.19, 0.34), (0.22, 0.44), (0.30, 0.44),
          (0.36, 0.50), (0.40, 0.56), (0.41, 0.30), (0.39, 0.02), (0.31, -0.10), (0.22, -0.16)]
    for pol in (pD, pE):
        R.poligono(R.ar, [T(*p) for p in suave(pol, 2)], 1.0)
    # traqueia
    R.poligono(R.ar, [T(*p) for p in [(-0.018, -0.34), (0.018, -0.34), (0.02, 0.06), (0.06, 0.12), (0.0, 0.10), (-0.06, 0.12), (-0.02, 0.06)]], 0.9)
    # mediastino, coração, arco aórtico, hilos
    R.poligono(R.sombra, [T(*p) for p in suave([(-0.07, -0.30), (0.07, -0.30), (0.09, 0.10), (-0.09, 0.10)], 2)], 0.25)
    R.poligono(R.sombra, [T(*p) for p in suave([(-0.08, 0.12), (0.02, 0.09), (0.14, 0.14), (0.22, 0.26), (0.27, 0.40),
                                              (0.22, 0.49), (0.04, 0.49), (-0.07, 0.42), (-0.10, 0.26)], 2)],
               0.42 if enfase == "coracao" else 0.32)
    R.elipse(R.sombra, T(0.10, 0.04), 0.05 * sw, 0.045 * sw, ang, 0.30)
    for s in (-1, 1):
        R.elipse(R.sombra, T(s * 0.13, 0.19), 0.045 * sw, 0.06 * sw, ang, 0.22)
        marcas_vasculares(R, c, "E" if s > 0 else "D", (s * 0.13, 0.19))
    R.elipse(R.ar, T(0.24, 0.55), 0.06 * sw, 0.04 * sw, ang, 0.8)  # bolha gástrica
    # clavículas
    for s in (-1, 1):
        cl = suave([(s * 0.05, -0.07), (s * 0.20, -0.13), (s * 0.35, -0.08), (s * 0.47, -0.03)], 2, fechado=False)
        pts = [T(*p) for p in cl]
        esq = [p + np.array([0, -0.017 * sw]) for p in pts]
        dd = [p + np.array([0, 0.017 * sw]) for p in pts]
        R.osso_poly(esq + dd[::-1], 0.38, 0.55)
        # escápula (translúcida) e cabeça do úmero
        esc = suave([(s * 0.28, 0.00), (s * 0.47, -0.04), (s * 0.44, 0.09), (s * 0.40, 0.22), (s * 0.34, 0.38),
                     (s * 0.30, 0.34), (s * 0.28, 0.18)], 2)
        R.osso_poly([T(*p) for p in esc], 0.16, 0.30)
        R.elipse(R.osso, T(s * 0.49, 0.05), 0.06 * sw, 0.06 * sw, ang, 0.45)
        R.elipse(R.cortex, T(s * 0.49, 0.05), 0.06 * sw, 0.06 * sw, ang, 0.45, esp_px=2.4)
        R.osso_longo(T(s * 0.49, 0.10), T(s * 0.54, 0.45), 0.050 * sw, 0.06 * sw, 0.06 * sw, gap=0.0, fill=0.40)
    costelas(R, c, 10, 1.35 if enfase == "costelas" else 1.0)
    coluna_frontal(R, c, ("toracica",))
    # esterno (sobreposto, discreto)
    R.osso_poly([T(*p) for p in [(-0.03, -0.08), (0.03, -0.08), (0.025, 0.34), (-0.025, 0.34)]], 0.12, 0.2)


def torax_lateral(R, c, enfase=None):
    sw, T = c.sw, c.tq
    ang = c.ang_ex
    R.poligono(R.ar, [T(*p) for p in suave([(-0.19, -0.12), (0.10, -0.15), (0.15, 0.0), (0.16, 0.45), (0.12, 0.62),
                                           (-0.02, 0.55), (-0.17, 0.46), (-0.20, 0.10)], 2)], 1.0)
    R.poligono(R.ar, [T(*p) for p in [(-0.012, -0.34), (0.012, -0.34), (0.012, -0.10), (-0.012, -0.10)]], 0.7)
    R.poligono(R.sombra, [T(*p) for p in suave([(-0.19, 0.12), (-0.04, 0.10), (0.05, 0.22), (0.07, 0.44),
                                              (-0.05, 0.52), (-0.19, 0.48)], 2)], 0.35)
    R.elipse(R.sombra, T(0.02, 0.16), 0.05 * sw, 0.06 * sw, ang, 0.25)
    for k in range(10):
        y0 = -0.02 + 0.058 * k
        for dx in (-0.015, 0.015):
            P0, P1, P2 = np.array([0.19 + dx, y0]), np.array([0.0 + dx, y0 + 0.02]), np.array([-0.20 + dx, y0 + 0.30])
            cur = [(1 - t) ** 2 * P0 + 2 * t * (1 - t) * P1 + t * t * P2 for t in np.linspace(0, 0.9, 20)]
            R.polilinha(R.osso, [T(*p) for p in cur], 0.024 * sw, 0.26)
    R.osso_poly([T(*p) for p in suave([(-0.235, -0.08), (-0.195, -0.08), (-0.18, 0.40), (-0.22, 0.40)], 2)], 0.45, 0.6)
    R.osso_poly([T(*p) for p in suave([(0.08, 0.0), (0.20, 0.02), (0.22, 0.28), (0.16, 0.36), (0.10, 0.20)], 2)], 0.14, 0.25)
    coluna_lateral(R, c, ("toracica", "lombar"), a0=0.22)
    for p0 in (0.40, 0.46):
        R.polilinha(R.sombra, [T(-0.18, p0 + 0.05), T(-0.05, p0 - 0.02), T(0.10, p0 + 0.04)], 0.012 * sw, 0.2)


# ---------------- pelve e abdome ----------------
def pelve_frontal(R, c, enfase=None):
    sw, T = c.sw, c.tq
    ang = c.ang_ex
    for s in (-1, 1):
        ilio = suave([(s * 0.07, 1.02), (s * 0.12, 0.92), (s * 0.26, 0.86), (s * 0.38, 0.93), (s * 0.41, 1.05),
                      (s * 0.36, 1.17), (s * 0.31, 1.20), (s * 0.27, 1.29), (s * 0.17, 1.30), (s * 0.10, 1.12)], 2)
        R.osso_poly([T(*p) for p in ilio], 0.20, 0.50)
        R.polilinha(R.cortex, [T(s * 0.10, 1.03), T(s * 0.17, 1.14), T(s * 0.21, 1.22)], 0.012 * sw, 0.40)
        pub = suave([(s * 0.17, 1.30), (s * 0.08, 1.35), (s * 0.03, 1.37), (s * 0.03, 1.46), (s * 0.08, 1.50),
                     (s * 0.15, 1.54), (s * 0.20, 1.50), (s * 0.20, 1.40), (s * 0.26, 1.32)], 2)
        R.osso_poly([T(*p) for p in pub], 0.34, 0.55)
        R.elipse(R.osso, T(s * 0.13, 1.44), 0.045 * sw, 0.07 * sw, ang, -0.30)  # forame obturado
        R.elipse(R.cortex, T(s * 0.22, 1.25), 0.095 * sw, 0.095 * sw, ang, 0.45, esp_px=2.6)  # acetábulo
    coluna_frontal(R, c, ("lombar", "sacro"))
    for s in (-1, 1):
        R.polilinha(R.cortex, [T(s * 0.09, 1.00), T(s * 0.115, 1.12), T(s * 0.10, 1.22)], 0.008 * sw, 0.25)


def abdome(R, c, modo="simples"):
    sw, T = c.sw, c.tq
    ang = c.ang_ex
    costelas(R, c, 12, 0.9)
    coluna_frontal(R, c, ("toracica", "lombar", "sacro"))
    pelve_frontal(R, c)
    pk = 0.45 if modo == "urinario" else 0.30
    for s in (-1, 1):
        R.poligono(R.sombra, [T(*p) for p in suave([(s * 0.04, 0.70), (s * 0.11, 0.72), (s * 0.20, 1.15), (s * 0.28, 1.42),
                                                  (s * 0.22, 1.45), (s * 0.14, 1.18), (s * 0.05, 0.92)], 2)],
                   0.22 if modo == "urinario" else 0.14)
        R.elipse(R.sombra, T(s * 0.17, 0.78), 0.055 * sw, 0.14 * sw, ang + s * 12, pk)
        if modo == "urinario":
            R.polilinha(R.sombra, [T(s * 0.13, 0.92), T(s * 0.10, 1.10), T(s * 0.08, 1.30), T(s * 0.04, 1.40)], 0.006 * sw, 0.10)
    R.poligono(R.sombra, [T(*p) for p in suave([(-0.40, 0.30), (-0.05, 0.32), (-0.04, 0.62), (-0.30, 0.72), (-0.40, 0.55)], 2)], 0.16)
    R.elipse(R.sombra, T(0, 1.44), 0.14 * sw, 0.09 * sw, ang, 0.13)
    R.elipse(R.ar, T(0.25, 0.52), 0.07 * sw, 0.05 * sw, ang, 0.7)
    # gás intestinal ao longo do cólon e alças
    caminho = [(-0.30, 1.10), (-0.32, 0.88), (-0.30, 0.72), (-0.15, 0.68), (0.0, 0.74), (0.15, 0.70),
               (0.30, 0.74), (0.32, 0.92), (0.30, 1.10), (0.22, 1.25), (0.08, 1.30)]
    for i in range(len(caminho) - 1):
        for t in np.linspace(0, 1, 3, endpoint=False):
            a = np.array(caminho[i]) * (1 - t) + np.array(caminho[i + 1]) * t
            if R.rng.random() < 0.7:
                R.elipse(R.ar, T(*a), R.rng.uniform(0.035, 0.06) * sw, R.rng.uniform(0.025, 0.045) * sw,
                         ang + R.rng.uniform(-30, 30), 0.55)


# ---------------- crânio ----------------
def cranio_frontal(R, c, enfase=None):
    sw = c.sw
    H = c.cab
    Lw = c.L
    ang = math.degrees(math.atan2(c.cab_x[1], c.cab_x[0]))
    vaso = suave([(0, -2.05), (0.65, -1.95), (1.05, -1.55), (1.2, -0.9), (1.2, -0.2), (1.12, 0.35), (0.95, 0.60),
                  (-0.95, 0.60), (-1.12, 0.35), (-1.2, -0.2), (-1.2, -0.9), (-1.05, -1.55), (-0.65, -1.95)], 2)
    R.osso_poly([H(*p) for p in vaso], 0.20, 0.8, 3.4)
    # ossos da face (maxila e zigomas)
    face = suave([(-1.0, 0.2), (-0.95, 0.75), (-0.7, 1.05), (0.7, 1.05), (0.95, 0.75), (1.0, 0.2),
                  (0.55, 0.45), (-0.55, 0.45)], 2)
    R.osso_poly([H(*p) for p in face], 0.20, 0.45)
    for s in (-1, 1):
        # órbitas
        R.elipse(R.cortex, H(s * 0.58, 0.0), 0.56 * Lw, 0.50 * Lw, ang, 0.75, esp_px=3.6)
        R.polilinha(R.osso, [H(s * 1.02, 0.52), H(s * 0.30, 0.10)], 0.22 * Lw, 0.28)  # rochedo
        # seios
        R.poligono(R.ar, [H(*p) for p in suave([(s * 0.34, 0.40), (s * 0.85, 0.42), (s * 0.95, 0.72), (s * 0.80, 0.98),
                                               (s * 0.45, 0.95), (s * 0.32, 0.70)], 2)], 0.8)
        R.elipse(R.ar, H(s * 0.55, -0.88), 0.42 * Lw, 0.20 * Lw, ang, 0.65)
        for k in range(3):
            R.elipse(R.ar, H(s * (0.18 + 0.05 * k), 0.02 + 0.13 * k), 0.07 * Lw, 0.08 * Lw, ang, 0.5)
        # arco zigomático
        R.polilinha(R.osso, [H(s * 1.0, 0.25), H(s * 1.12, 0.6), H(s * 0.95, 0.95)], 0.09 * Lw, 0.45)
        # mastoide com células aéreas
        for k in range(8):
            R.elipse(R.ar, H(s * (1.10 + R.rng.normal(0, 0.03)), 0.45 + 0.07 * k), 0.045 * Lw, 0.045 * Lw, 0, 0.45)
    # cavidade nasal e septo
    R.poligono(R.ar, [H(*p) for p in suave([(-0.10, 0.25), (0.10, 0.25), (0.24, 0.75), (0.22, 0.95), (-0.22, 0.95), (-0.24, 0.75)], 2)], 0.7)
    R.linha(R.osso, H(0, 0.3), H(0, 0.95), 0.05 * Lw, 0.35)
    # dentes
    for i in range(-3, 4):
        for (y0, y1) in ((0.98, 1.16), (1.18, 1.34)):
            R.osso_poly([H(i * 0.19 - 0.08, y0), H(i * 0.19 + 0.08, y0), H(i * 0.19 + 0.07, y1), H(i * 0.19 - 0.07, y1)], 0.55, 0.75, 1.6)
    # mandíbula
    mand = [H(*p) for p in suave([(-1.02, 0.55), (-1.05, 1.2), (-0.92, 1.55), (-0.45, 1.78), (0.45, 1.78), (0.92, 1.55),
                                 (1.05, 1.2), (1.02, 0.55)], 2, fechado=False)]
    R.polilinha(R.osso, mand, 0.16 * Lw, 0.40)
    R.polilinha(R.cortex, mand, 0.03 * Lw, 0.40)


def cranio_lateral(R, c, enfase=None):
    Lw = c.L
    H = c.cab
    ang = math.degrees(math.atan2(c.cab_x[1], c.cab_x[0]))
    cranio = suave([(-1.45, -0.55), (-1.25, -1.35), (-0.55, -1.95), (0.5, -2.1), (1.25, -1.7), (1.62, -0.9),
                    (1.55, 0.0), (1.20, 0.55), (0.3, 0.62), (-0.5, 0.55), (-1.0, 0.45), (-1.5, 0.20)], 2)
    R.osso_poly([H(*p) for p in cranio], 0.20, 0.8, 3.4)
    R.osso_poly([H(*p) for p in suave([(-1.55, 0.2), (-1.85, 0.62), (-1.6, 0.75), (-1.5, 1.05), (-0.5, 1.05), (-0.35, 0.6)], 2)], 0.22, 0.55)
    R.elipse(R.ar, H(-1.25, -0.55), 0.36 * Lw, 0.2 * Lw, ang, 0.65)             # seio frontal
    R.elipse(R.ar, H(-1.02, 0.66), 0.45 * Lw, 0.27 * Lw, ang, 0.8)               # seio maxilar
    R.poligono(R.ar, [H(*p) for p in suave([(-1.55, 0.3), (-0.45, 0.4), (-0.4, 0.55), (-1.45, 0.62)], 2)], 0.6)  # fossa nasal
    R.elipse(R.ar, H(-0.38, 0.38), 0.28 * Lw, 0.17 * Lw, ang, 0.75)              # seio esfenoidal
    R.polilinha(R.cortex, [H(-0.12, 0.05), H(-0.18, 0.22), H(-0.34, 0.26), H(-0.50, 0.14)], 0.05 * Lw, 0.8)  # sela
    R.polilinha(R.cortex, [H(-0.10, -0.10), H(-0.06, 0.18)], 0.07 * Lw, 0.8)     # dorso da sela
    R.poligono(R.ar, [H(*p) for p in suave([(-0.55, 0.62), (-0.28, 0.62), (-0.18, 1.25), (-0.40, 1.28)], 2)], 0.7)  # cavum
    R.elipse(R.sombra, H(-0.35, 0.58), 0.22 * Lw, 0.10 * Lw, ang, 0.30)          # adenoides
    R.elipse(R.cortex, H(0.45, 0.50), 0.08 * Lw, 0.08 * Lw, ang, 0.6, esp_px=2.4)  # meato acústico
    R.elipse(R.osso, H(0.7, 0.45), 0.45 * Lw, 0.25 * Lw, ang, 0.28)              # rochedo
    for k in range(10):
        R.elipse(R.ar, H(0.95 + R.rng.normal(0, 0.05), 0.55 + 0.045 * k), 0.04 * Lw, 0.04 * Lw, 0, 0.45)  # mastoide
    for i in range(5):
        for (y0, y1) in ((0.98, 1.16), (1.18, 1.34)):
            R.osso_poly([H(-1.45 + i * 0.1, y0), H(-1.36 + i * 0.1, y0), H(-1.37 + i * 0.1, y1), H(-1.46 + i * 0.1, y1)], 0.5, 0.7, 1.4)
    mand = suave([(0.15, 0.55), (0.35, 0.9), (0.55, 1.5), (0.35, 1.75), (-0.2, 1.78), (-0.9, 1.7), (-1.35, 1.55),
                  (-1.45, 1.38), (-1.2, 1.36), (-0.5, 1.35), (-0.05, 0.9), (-0.15, 0.6)], 2)
    R.osso_poly([H(*p) for p in mand], 0.34, 0.6)
    R.elipse(R.osso, H(0.12, 0.58), 0.1 * Lw, 0.14 * Lw, ang, 0.4)               # côndilo (ATM)


# ---------------- membros ----------------
def lateral_vec(c, lado, eixo):
    n = np.array([-eixo[1], eixo[0]])
    return n if (n[0] > 0) == (lado == "E") else -n


def mao_ossos(R, c, lado):
    H = c.mao[lado]
    Lm = float(np.linalg.norm(H[12] - H[0]))
    u = unit(H[9] - H[0])
    perp = np.array([-u[1], u[0]])
    for i in range(2):  # carpo
        for j in range(4):
            p = H[0] + u * Lm * (0.10 + 0.085 * i) + perp * Lm * (j - 1.5) * 0.085
            R.osso_poly(suave([p + perp * Lm * 0.04 + u * Lm * 0.035, p - perp * Lm * 0.04 + u * Lm * 0.035,
                               p - perp * Lm * 0.04 - u * Lm * 0.035, p + perp * Lm * 0.04 - u * Lm * 0.035], 2), 0.55, 0.6, 1.8)
    dedos = {"polegar": (1, 2, 3, 4), "indicador": (5, 6, 7, 8), "medio": (9, 10, 11, 12),
             "anelar": (13, 14, 15, 16), "mindinho": (17, 18, 19, 20)}
    for nome, (a, b, cc, d) in dedos.items():
        base = H[a] if nome == "polegar" else lerp_pt(H[0], H[a], 0.30)
        for p, q, fator in ((base, H[b], 0.15), (H[b], H[cc], 0.20), (H[cc], H[d], 0.20)):
            L = float(np.linalg.norm(q - p))
            if nome == "polegar" and p is base:
                pass
            R.osso_longo(p, q, fator * L * 0.85, fator * L * 1.25, fator * L * 1.25, gap=0.06, fill=0.55, cort=0.7, canal=False)
        # falange distal e média
    return


def lerp_pt(a, b, t):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t


def membro_sup(R, c, lado, mao=True):
    sw = c.sw
    P = c.P
    sh, el, wr = P["ombro_" + lado], P["cotovelo_" + lado], P["punho_" + lado]
    s_mao = c.sw_mao[lado]
    u_um = unit(el - sh)
    R.elipse(R.osso, sh, 0.058 * sw, 0.058 * sw, 0, 0.45)
    R.osso_longo(sh + u_um * 0.04 * sw, el, 0.050 * sw, 0.07 * sw, 0.13 * sw, gap=0.02)
    u = unit(wr - el)
    lat = lateral_vec(c, lado, u)
    k = 0.045 * s_mao * 1.0
    R.osso_longo(el + lat * k * 0.6, wr + lat * k * 1.1, 0.034 * s_mao, 0.050 * s_mao, 0.085 * s_mao, gap=0.03)  # rádio
    R.osso_longo(el - lat * k * 0.9, wr - lat * k * 1.5, 0.034 * s_mao, 0.085 * s_mao, 0.048 * s_mao, gap=0.03)  # ulna
    if mao:
        mao_ossos(R, c, lado)


def pe_ossos(R, c, lado):
    sw = c.sw
    P = c.P
    an, hl, tp = P["tornozelo_" + lado], P["calcanhar_" + lado], P["ponta_pe_" + lado]
    jo = P["joelho_" + lado]
    perna = unit(an - jo)
    if np.linalg.norm(tp - hl) < 0.35 * sw:  # pé de frente para a câmera: usa o eixo da perna
        hl = an + perna * 0.12 * sw
        tp = hl + perna * 0.62 * sw
    f = unit(tp - hl)
    Lf = float(np.linalg.norm(tp - hl))
    lat = lateral_vec(c, lado, f)
    ang = math.degrees(math.atan2(f[1], f[0]))
    pt = lambda t, off: hl + f * Lf * t + lat * Lf * off
    R.elipse(R.osso, pt(0.04, 0.0), 0.15 * Lf, 0.11 * Lf, ang, 0.55)                 # calcâneo
    R.elipse(R.cortex, pt(0.04, 0.0), 0.15 * Lf, 0.11 * Lf, ang, 0.55, esp_px=2.2)
    R.elipse(R.osso, pt(0.20, 0.0), 0.10 * Lf, 0.10 * Lf, ang, 0.5)                  # tálus
    for off in (-0.09, 0.0, 0.09):
        R.elipse(R.osso, pt(0.36, off), 0.05 * Lf, 0.045 * Lf, ang, 0.5)
    mts = [(-0.14, -0.15), (-0.07, -0.075), (0.0, 0.0), (0.06, 0.07), (0.115, 0.135)]
    for i, (ob, oh) in enumerate(mts):
        R.osso_longo(pt(0.44, ob), pt(0.76, oh), 0.040 * Lf * (1.5 if i == 0 else 1), 0.06 * Lf, 0.07 * Lf, gap=0.04, fill=0.55)
        fim = 1.0 if i < 2 else 0.95 - 0.02 * i
        segs = [(0.76, 0.87)] if i == 0 else [(0.76, 0.83), (0.83, 0.90)]
        for t0, t1 in segs:
            R.osso_longo(pt(t0, oh * 1.02), pt(t1, oh * 1.03), 0.038 * Lf, 0.05 * Lf, 0.05 * Lf, gap=0.08, fill=0.55, canal=False)
        R.osso_longo(pt(segs[-1][1], oh * 1.03), pt(fim, oh * 1.04), 0.030 * Lf, 0.04 * Lf, 0.03 * Lf, gap=0.08, fill=0.55, canal=False)


def membro_inf(R, c, lado, pe=True):
    sw = c.sw
    P = c.P
    hip, kn, an = P["quadril_" + lado], P["joelho_" + lado], P["tornozelo_" + lado]
    u = unit(kn - hip)
    lat = lateral_vec(c, lado, u)
    R.elipse(R.osso, hip, 0.062 * sw, 0.062 * sw, 0, 0.55)                             # cabeça do fêmur
    R.elipse(R.cortex, hip, 0.062 * sw, 0.062 * sw, 0, 0.55, esp_px=2.4)
    troc = hip + lat * 0.17 * sw + u * 0.10 * sw
    R.osso_longo(hip + lat * 0.02 * sw, troc, 0.060 * sw, 0.05 * sw, 0.10 * sw, gap=0.0, canal=False)
    R.osso_longo(hip + lat * 0.13 * sw + u * 0.17 * sw, kn - u * 0.015 * sw, 0.075 * sw, 0.11 * sw, 0.19 * sw, gap=0.0)
    R.elipse(R.osso, kn - u * 0.08 * sw, 0.036 * sw, 0.045 * sw, math.degrees(math.atan2(u[1], u[0])) + 90, 0.30)  # patela
    u2 = unit(an - kn)
    med = -lateral_vec(c, lado, u2)
    R.osso_longo(kn + u2 * 0.03 * sw + med * 0.02 * sw, an + med * 0.02 * sw, 0.062 * sw, 0.16 * sw, 0.10 * sw, gap=0.02)  # tíbia
    lat2 = -med
    R.osso_longo(kn + u2 * 0.07 * sw + lat2 * 0.10 * sw, an + lat2 * 0.075 * sw + u2 * 0.03 * sw, 0.030 * sw, 0.05 * sw, 0.055 * sw, gap=0.02)  # fíbula
    if pe:
        pe_ossos(R, c, lado)


# ----------------------------------------------------------------------
# Planejamento por exame
# ----------------------------------------------------------------------
def planejar(nome, incid, c):
    """Define o que desenhar e o enquadramento de cada exame da lista."""
    n = remover_acentos(nome).upper().strip()
    lado = "E" if " ESQ" in n else "D" if " DIR" in n else None
    lat = ("PERFIL" in n or "DINAMICA" in n or any(k in n for k in ("SELA", "CAVUM", "TEMPOROMANDIBULAR")))
    suporta = any(k in n for k in ("TORAX", "COLUNA", "CRANIO", "SEIOS", "MAXILAR", "OSSOS DA FACE", "ESTERNO",
                                   "MANDIBULA", "ORBITAS", "MASTOIDES", "ARCADA", "SACRO-COXIGEA"))
    if incid == "2" and suporta:
        lat = True
    flex = 0.0
    if "DINAMICA" in n:
        flex = {"1": 0.55, "2": -0.45}.get(incid, 0.0)
    ls = [lado] if lado else ["D", "E"]
    ladoref = lado or "D"
    P = c.P

    def roi_tronco(a0, a1, b0, b1, **kw):
        pts = [c.tq(a0, b0), c.tq(a1, b0), c.tq(a0, b1), c.tq(a1, b1)]
        return dict(pts=pts, eixo=(c.tq(0, b1), c.tq(0, b0)), **kw)

    def roi_ponto(p, hw, hh, eixo):
        return dict(centro=p, meia=(hw, hh), eixo=eixo)

    D = []
    marca = None
    sil_lat = False
    sw = c.sw

    # --- Tórax ---
    if any(k in n for k in ("TORAX", "COSTELAS", "HEMITORAX", "ESTERNO", "CORACAO")):
        if lat or "ESTERNO" in n:
            c.modo_lateral()
            sil_lat = True
            D = [(torax_lateral, {})]
            roi = roi_tronco(-0.45, 0.45, -0.30, 0.75) if "ESTERNO" not in n else roi_tronco(-0.40, 0.10, -0.15, 0.50)
        else:
            enf = "costelas" if "COSTELAS" in n else "coracao" if "CORACAO" in n else None
            D = [(torax_frontal, {"enfase": enf})]
            roi = roi_tronco(-0.60, 0.60, -0.38, 0.88)
        marca = ("D", "esq") if not (lat or "ESTERNO" in n) else None
    # --- Coluna ---
    elif "COLUNA" in n or "SACRO-COXIGEA" in n:
        trechos = ("cervical", "toracica", "lombar", "sacro")
        if "CERVICAL" in n:
            tr, b0, b1 = ("cervical",), -0.62, 0.12
        elif "DORSAL" in n:
            tr, b0, b1 = ("toracica",), -0.15, 0.80
        elif "LOMBO" in n:
            tr, b0, b1 = ("lombar", "sacro"), 0.55, 1.50
        elif "LOMBAR" in n:
            tr, b0, b1 = ("lombar", "sacro"), 0.55, 1.35
        elif "SACRO" in n:
            tr, b0, b1 = ("lombar", "sacro"), 0.95, 1.55
        else:
            tr, b0, b1 = trechos, -0.75, 1.55
        if lat:
            c.modo_lateral()
            sil_lat = True
            D = [(coluna_lateral, {"trechos": tr, "flex": flex})]
            roi = roi_tronco(-0.30, 0.35, b0, b1)
        else:
            if "TOTAL" in n:
                D = [(torax_frontal, {}), (pelve_frontal, {})]
                roi = roi_tronco(-0.60, 0.60, b0, b1)
            else:
                D = [(coluna_frontal, {"trechos": tr})]
                if "toracica" in tr:
                    D = [(torax_frontal, {})]
                if "lombar" in tr:
                    D = [(pelve_frontal, {}), (coluna_frontal, {"trechos": tr})]
                roi = roi_tronco(-0.40, 0.40, b0, b1)
            marca = ("D", "esq")
    # --- Crânio e face ---
    elif any(k in n for k in ("CRANIO", "FACE", "SEIOS", "CAVUM", "MAXILAR", "MANDIBULA", "ZIGOMATICA",
                              "TEMPOROMANDIBULAR", "SELA", "ORBITAS", "MASTOIDES")):
        if "CAVUM" in n or "SELA" in n or "TEMPOROMANDIBULAR" in n:
            lat = True
        c.lateral = True if lat else c.lateral
        sil_lat = lat
        D = [(cranio_lateral if lat else cranio_frontal, {})]
        e = (c.cab(0, 1.9), c.cab(0, -2.1))
        recortes = {"SEIOS": (0, -0.1, 1.5, 1.3), "OSSOS DA FACE": (0, 0.2, 1.4, 1.5), "MAXILAR": (0, 0.7, 1.1, 0.7),
                    "MANDIBULA": (0, 1.15, 1.25, 0.85), "ZIGOMATICA": (0, 0.45, 1.5, 0.75),
                    "TEMPOROMANDIBULAR": (0.2, 0.65, 0.8, 0.8), "SELA": (-0.2, 0.2, 0.7, 0.7),
                    "CAVUM": (-0.4, 0.75, 0.8, 0.8), "ORBITAS": (0, 0.0, 1.0, 0.7),
                    "MASTOIDES": (0.5 if lat else 0.0, 0.45, 0.8 if lat else 1.5, 0.8)}
        rc = next((v for k, v in recortes.items() if k in n), (0, -0.1, 1.7, 2.2))
        roi = roi_ponto(c.cab(rc[0], rc[1]), rc[2] * c.L, rc[3] * c.L, e)
        marca = ("D", "esq") if not lat else None
    # --- Membro superior ---
    elif any(k in n for k in ("MEMBRO SUPERIOR", "OMBRO", "ACROMIO", "ESCAPULA", "CLAVICULA", "BRACO", "COTOVELO",
                              "PUNHO", "MAO", "DEDOS")):
        if "MAOS E PUNHOS" in n:
            D = [(membro_sup, {"lado": l_}) for l_ in ("D", "E")]
            pts = [P["punho_D"], P["punho_E"]] + [c.mao[l_][12] for l_ in ("D", "E")] + [c.mao[l_][4] for l_ in ("D", "E")]
            ctr_p = (P["punho_D"] + P["punho_E"]) / 2
            ctr_d = (c.mao["D"][12] + c.mao["E"][12]) / 2
            roi = dict(pts=pts, eixo=(ctr_p, ctr_d))
            marca = ("D", "esq")
        else:
            l_ = ladoref
            sh, el, wr = P["ombro_" + l_], P["cotovelo_" + l_], P["punho_" + l_]
            H = c.mao[l_]
            sm = c.sw_mao[l_]
            D = [(membro_sup, {"lado": l_})]
            if any(k in n for k in ("OMBRO", "ACROMIO", "ESCAPULA", "CLAVICULA")):
                D = [(torax_frontal, {}), (membro_sup, {"lado": l_, "mao": False})]
                s = 1 if l_ == "E" else -1
                if "CLAVICULA" in n:
                    roi = dict(centro=c.tq(s * 0.27, -0.06), meia=(0.30 * sw, 0.22 * sw), eixo=(c.tq(0, 1), c.tq(0, 0)))
                elif "ACROMIO" in n:
                    roi = dict(centro=c.tq(s * 0.45, 0.0), meia=(0.20 * sw, 0.20 * sw), eixo=(c.tq(0, 1), c.tq(0, 0)))
                elif "ESCAPULA" in n:
                    roi = dict(centro=c.tq(s * 0.36, 0.20), meia=(0.30 * sw, 0.30 * sw), eixo=(c.tq(0, 1), c.tq(0, 0)))
                else:
                    roi = dict(centro=c.tq(s * 0.42, 0.10), meia=(0.40 * sw, 0.40 * sw), eixo=(c.tq(0, 1), c.tq(0, 0)))
            elif "MEMBRO SUPERIOR" in n:
                D = [(membro_sup, {"lado": l_})]
                roi = dict(pts=[sh, el, wr, H[12], H[4]], eixo=(H[12], sh))
            elif "ANTEBRACO" in n:
                roi = dict(pts=[el, wr], eixo=(wr, el))
            elif "COTOVELO" in n:
                roi = roi_ponto(el, 0.42 * sw, 0.42 * sw, (wr, sh))
            elif "BRACO" in n:
                roi = dict(pts=[sh, el], eixo=(el, sh))
            elif "PUNHO" in n:
                roi = roi_ponto(wr, 0.26 * sm, 0.30 * sm, (wr, el))
            elif "DEDOS" in n:
                roi = dict(pts=[H[5], H[9], H[17], H[8], H[12], H[16], H[20], H[4]], eixo=(H[0], H[12]))
            else:  # MAO
                roi = dict(pts=[H[0], H[4], H[8], H[12], H[16], H[20]], eixo=(H[0], H[12]))
            marca = (l_, "esq" if l_ == "D" else "dir")
    # --- Abdome ---
    elif "ABDOMEN" in n or "VIAS URINARIAS" in n:
        modo = "urinario" if "URINARIAS" in n else "simples"
        D = [(abdome, {"modo": modo})]
        roi = roi_tronco(-0.55, 0.55, 0.20 if "AGUDO" in n else 0.55, 1.62)
        marca = ("D", "esq")
    # --- Pelve e membro inferior ---
    else:
        if "ESCANOMETRIA" in n:
            D = [(pelve_frontal, {})] + [(membro_inf, {"lado": l_}) for l_ in ("D", "E")]
            pts = [P["quadril_D"], P["quadril_E"], P["tornozelo_D"], P["tornozelo_E"], P["ponta_pe_D"], P["ponta_pe_E"]]
            roi = dict(pts=pts, eixo=((P["tornozelo_D"] + P["tornozelo_E"]) / 2, (P["quadril_D"] + P["quadril_E"]) / 2))
            marca = ("D", "esq")
        elif any(k in n for k in ("BACIA", "SACROILIACA")):
            D = [(pelve_frontal, {})] + [(membro_inf, {"lado": l_, "pe": False}) for l_ in ("D", "E")]
            if "SACROILIACA" in n:
                roi = roi_tronco(-0.30, 0.30, 0.85, 1.40)
            else:
                roi = roi_tronco(-0.55, 0.55, 0.80, 1.85)
            marca = ("D", "esq")
        else:
            l_ = ladoref
            hip, kn, an = P["quadril_" + l_], P["joelho_" + l_], P["tornozelo_" + l_]
            hl, tp = P["calcanhar_" + l_], P["ponta_pe_" + l_]
            D = [(pelve_frontal, {}), (membro_inf, {"lado": l_})]
            ax = (kn, hip)
            if "MEMBRO INFERIOR" in n:
                roi = dict(pts=[hip, kn, an, tp], eixo=(tp, hip))
            elif "COXOFEMORAL" in n:
                roi = roi_ponto(hip, 0.55 * sw, 0.55 * sw, (kn, hip))
            elif "COXA" in n:
                roi = dict(pts=[hip, kn], eixo=ax)
            elif "JOELHO" in n or "PATELA" in n:
                roi = roi_ponto(kn, 0.40 * sw, 0.40 * sw, (an, hip))
            elif "PERNA" in n:
                roi = dict(pts=[kn, an], eixo=(an, kn))
            elif "TORNOZELO" in n:
                roi = roi_ponto(an, 0.32 * sw, 0.32 * sw, (tp, an))
            elif "CALCANEO" in n:
                roi = roi_ponto(hl, 0.26 * sw, 0.26 * sw, (tp, an))
            else:  # PE
                roi = dict(pts=[an, hl, tp], eixo=(an, tp))
            marca = (l_, "esq" if l_ == "D" else "dir")
    return dict(desenhos=D, roi=roi, lateral=sil_lat, marca=marca)


def gerar_radiografia(exame, incidencia, largura, altura, pose=None, maos=None, semente=7):
    """Gera a radiografia simulada (imagem em tons de cinza, uint8) para o exame escolhido."""
    c = Corpo(largura, altura, pose, maos)
    plano = planejar(exame, str(incidencia), c)
    roi = dict(plano["roi"])
    R = Render(roi.pop("eixo"), saida=1100, semente=semente, **roi)
    silhueta(R, c, lateral=plano["lateral"])
    for fn, kw in plano["desenhos"]:
        fn(R, c, **kw)
    marca = plano["marca"]
    return R.finalizar(marca)