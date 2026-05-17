"""
pipeline.py
===========
Orquesta el pipeline completo de generación de subtítulos:
  1. Extracción de audio (ffmpeg, 3 estrategias)
  2. Separación vocal opcional (Demucs, para videos musicales)
  3. Transcripción ASR (faster-whisper, CPU int8 por defecto)
  4. Corrección LLM (vía llama-server HTTP)
  5. Traducción LLM opcional
  6. Formateo profesional (CPL/CPS/gap/pirámide)
  7. Exportación .srt
"""

import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from llm_client import LlamaServerClient, correct_segments, translate_segments
from srt_formatter import format_segments

log = logging.getLogger(__name__)


# ── Estructuras de datos ───────────────────────────────────────────────────────

@dataclass
class Segment:
    index: int
    start: float
    end:   float
    text:  str


@dataclass
class PipelineOptions:
    language:        str  = "auto"
    whisper_model:   str  = "auto"   # "auto" → resolve_whisper_config() elige
    use_demucs:      bool = False
    correct_with_llm: bool = True
    translate_to:    Optional[str] = None
    whisper_device:  str  = "auto"   # "auto" → resuelto en runtime
    whisper_compute: str  = "auto"   # "auto" → resuelto en runtime


def resolve_whisper_config(opts: "PipelineOptions") -> tuple[str, str, str]:
    """
    Devuelve (model, device, compute_type) óptimos para el hardware actual.

    Lógica:
      - Consulta VRAM libre vía nvidia-smi
      - Si GPU libre > 4 GB → GPU float16 + large-v3   (máxima calidad, rápido)
      - Si GPU libre 2-4 GB → GPU float16 + medium
      - Si GPU libre < 2 GB o sin GPU → CPU int8 + medium  (siempre funciona)
    El modelo explícito del usuario (no "auto") siempre tiene prioridad.
    """
    # Si el usuario eligió modelo/device explícitamente, respetarlo
    model   = opts.whisper_model   if opts.whisper_model   != "auto" else None
    device  = opts.whisper_device  if opts.whisper_device  != "auto" else None
    compute = opts.whisper_compute if opts.whisper_compute != "auto" else None

    if model and device and compute:
        return model, device, compute

    # Auto-detectar VRAM libre
    free_vram_gb = 0.0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            free_vram_gb = int(result.stdout.strip().split("\n")[0]) / 1024
    except Exception:
        pass

    log.info(f"[Whisper] VRAM libre: {free_vram_gb:.1f} GB")

    if free_vram_gb >= 4.0:
        resolved_device  = device  or "cuda"
        resolved_compute = compute or "float16"
        resolved_model   = model   or "large-v3"
        log.info("[Whisper] Auto: GPU large-v3 float16 (GPU disponible)")
    elif free_vram_gb >= 2.0:
        resolved_device  = device  or "cuda"
        resolved_compute = compute or "float16"
        resolved_model   = model   or "medium"
        log.info("[Whisper] Auto: GPU medium float16 (VRAM limitada)")
    else:
        resolved_device  = device  or "cpu"
        resolved_compute = compute or "int8"
        resolved_model   = model   or "medium"
        log.info("[Whisper] Auto: CPU medium int8 (GPU no disponible/ocupada)")

    return resolved_model, resolved_device, resolved_compute


# ── Utilidades SRT ─────────────────────────────────────────────────────────────

def _ts(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def segments_to_srt(segments: list[Segment]) -> str:
    blocks = []
    for seg in segments:
        blocks.append(f"{seg.index}\n{_ts(seg.start)} --> {_ts(seg.end)}\n{seg.text}\n")
    return "\n".join(blocks)


# ── Etapa 1: extracción de audio ───────────────────────────────────────────────

def _run_ffmpeg(cmd: list, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def extract_audio(input_path: str, output_path: str) -> str:
    """Convierte cualquier formato a WAV mono 16 kHz (para Whisper)."""
    flags = ["-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", "-y"]

    strategies = [
        ["ffmpeg", "-i", input_path] + flags + [output_path],
        ["ffmpeg", "-i", input_path, "-map", "0:a:0"] + flags + [output_path],
        ["ffmpeg", "-fflags", "+genpts+igndts", "-err_detect", "ignore_err",
         "-i", input_path, "-map", "0:a:0"] + flags + [output_path],
    ]

    for i, cmd in enumerate(strategies, 1):
        result = _run_ffmpeg(cmd)
        if result.returncode == 0:
            log.info(f"[Audio] Extraído con estrategia {i}: {output_path}")
            return output_path
        log.warning(f"[Audio] Estrategia {i} falló")

    stderr = result.stderr.decode(errors="replace")
    raise RuntimeError(
        f"ffmpeg no pudo extraer el audio. El archivo puede estar corrupto.\n"
        f"Detalle: {stderr[-400:]}"
    )


def extract_audio_hq(input_path: str, output_path: str) -> str:
    """Extrae audio estéreo 44100 Hz para Demucs (requiere mayor calidad)."""
    flags = ["-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", "-y"]
    strategies = [
        ["ffmpeg", "-i", input_path] + flags + [output_path],
        ["ffmpeg", "-i", input_path, "-map", "0:a:0"] + flags + [output_path],
    ]
    for i, cmd in enumerate(strategies, 1):
        result = _run_ffmpeg(cmd)
        if result.returncode == 0:
            log.info(f"[Audio HQ] Extraído con estrategia {i}: {output_path}")
            return output_path
    log.warning("[Audio HQ] Falló extracción HQ, usando 16kHz como fallback")
    return extract_audio(input_path, output_path)


# ── Etapa 2: separación vocal con Demucs ──────────────────────────────────────

def separate_vocals(wav_path: str, work_dir: str) -> str:
    """
    Separa la voz del fondo musical usando Demucs.
    Devuelve la ruta al archivo vocals.wav resultante.
    Requiere: pip install demucs
    """
    log.info("[Demucs] Separando vocales (puede tardar varios minutos)...")

    cmd = [
        sys.executable, "-m", "demucs",  # mismo Python del venv
        "--two-stems", "vocals",
        "--out", work_dir,
        "--device", "cpu",      # CPU para no competir con llama-server por VRAM
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=1800)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        log.warning(f"[Demucs] Error: {err[-300:]}. Usando audio original.")
        return wav_path

    # Demucs crea: {work_dir}/htdemucs/{stem}/vocals.wav
    # Salida WAV (sin --mp3) evita el encoder delay del MP3 que desincroniza
    vocals = next(Path(work_dir).rglob("vocals.wav"), None)
    if vocals and vocals.exists():
        size_mb = vocals.stat().st_size / 1024 / 1024
        log.info(f"[Demucs] Vocales separadas: {vocals} ({size_mb:.1f} MB)")

        # Convertir a WAV mono 16 kHz para Whisper
        vocals_wav = str(Path(work_dir) / "vocals_16k.wav")
        r = subprocess.run(
            ["ffmpeg", "-i", str(vocals), "-ac", "1", "-ar", "16000",
             "-acodec", "pcm_s16le", "-y", vocals_wav],
            capture_output=True, timeout=120,
        )
        if r.returncode == 0 and Path(vocals_wav).exists():
            return vocals_wav
        log.warning(f"[Demucs] ffmpeg falló al convertir vocales: "
                    f"{r.stderr.decode(errors='replace')[-200:]}")

    log.warning("[Demucs] No se encontró vocals.wav. Usando audio original.")
    return wav_path


# ── Etapa 3: transcripción ASR ─────────────────────────────────────────────────

def transcribe(audio_path: str, opts: PipelineOptions) -> tuple[list[Segment], dict]:
    """
    Transcribe con stable-ts (Whisper + refinamiento de timestamps por silencio/energía).
    stable-ts resuelve el problema de timestamps tempranos en música ajustando
    cada segmento al inicio real del sonido, no al inicio del chunk de Whisper.
    Fallback a faster-whisper si stable-ts falla.
    """
    try:
        return _transcribe_stable(audio_path, opts)
    except Exception as e:
        log.warning(f"[stable-ts] Error ({e}), usando faster-whisper como fallback")
        return _transcribe_faster(audio_path, opts)


def _transcribe_stable(audio_path: str, opts: PipelineOptions) -> tuple[list[Segment], dict]:
    import stable_whisper

    w_model, w_device, w_compute = resolve_whisper_config(opts)
    log.info(f"[stable-ts] Modelo '{w_model}' en {w_device}/{w_compute}")

    lang = None if opts.language == "auto" else opts.language
    is_music = opts.use_demucs

    model = stable_whisper.load_faster_whisper(w_model, device=w_device,
                                               compute_type=w_compute)
    result = model.transcribe_stable(
        audio_path,
        language=lang,
        beam_size=5,
        # suppress_silence=True ajusta cada segmento al inicio real del audio,
        # eliminando el offset que Whisper introduce con silencios/música antes de la voz
        suppress_silence=True,
        suppress_word_ts=False,
        vad=is_music,           # VAD adicional de stable-ts para música
        regroup=True,           # reagrupa frases de forma natural
    )

    detected_lang = result.language or (lang or "en")
    log.info(f"[stable-ts] Idioma detectado: {detected_lang}")

    segments = []
    for i, seg in enumerate(result.segments, 1):
        text = seg.text.strip()
        if not text:
            continue
        segments.append(Segment(index=i, start=round(seg.start, 3),
                                end=round(seg.end, 3), text=text))

    log.info(f"[stable-ts] {len(segments)} segmentos")

    audio_duration = 0.0
    try:
        import soundfile as sf
        with sf.SoundFile(audio_path) as f:
            audio_duration = len(f) / f.samplerate
    except Exception:
        pass

    return segments, {
        "language":             detected_lang,
        "language_probability": 1.0,
        "duration":             audio_duration,
        "whisper_model":        w_model,
        "whisper_device":       w_device,
    }


def _transcribe_faster(audio_path: str, opts: PipelineOptions) -> tuple[list[Segment], dict]:
    """Fallback: faster-whisper con VAD."""
    from faster_whisper import WhisperModel

    w_model, w_device, w_compute = resolve_whisper_config(opts)
    log.info(f"[Whisper-fallback] Modelo '{w_model}' en {w_device}/{w_compute}")
    model = WhisperModel(w_model, device=w_device, compute_type=w_compute)

    lang = None if opts.language == "auto" else opts.language
    raw_segs, info = model.transcribe(
        audio_path, language=lang, beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=False,
    )
    segments = []
    for i, seg in enumerate(raw_segs, 1):
        text = seg.text.strip()
        if text:
            segments.append(Segment(index=i, start=seg.start, end=seg.end, text=text))

    if not segments:
        raw_segs2, info = model.transcribe(audio_path, language=lang,
                                           beam_size=5, vad_filter=False,
                                           word_timestamps=False)
        for i, seg in enumerate(raw_segs2, 1):
            text = seg.text.strip()
            if text:
                segments.append(Segment(index=i, start=seg.start,
                                        end=seg.end, text=text))

    detected_lang = info.language
    log.info(f"[Whisper-fallback] {len(segments)} segmentos. Idioma: {detected_lang}")
    return segments, {
        "language":             detected_lang,
        "language_probability": round(info.language_probability, 3),
        "duration":             getattr(info, "duration", 0),
        "whisper_model":        w_model,
        "whisper_device":       w_device,
    }


# ── Pipeline principal ─────────────────────────────────────────────────────────

class PipelineRunner:
    def __init__(self, opts: PipelineOptions, llm: LlamaServerClient,
                 on_progress: Optional[Callable[[str, int], None]] = None):
        self._opts = opts
        self._llm  = llm
        self._prog = on_progress or (lambda step, pct: None)

    def _step(self, name: str, pct: int):
        log.info(f"[Pipeline] {pct}% — {name}")
        self._prog(name, pct)

    def run(self, input_path: str) -> tuple[list[Segment], dict]:
        opts = self._opts

        with tempfile.TemporaryDirectory(prefix="vsub_") as tmp:
            wav = str(Path(tmp) / "audio.wav")

            # 1. Extraer audio
            self._step("Extrayendo audio", 10)
            if opts.use_demucs:
                # Demucs necesita 44100 Hz estéreo para una separación precisa.
                # Con 16 kHz mono la separación es peor y filtra mal los instrumentos.
                audio_hq = str(Path(tmp) / "audio_hq.wav")
                extract_audio_hq(input_path, audio_hq)
                self._step("Separando vocales (Demucs)", 25)
                wav = separate_vocals(audio_hq, tmp)
            else:
                extract_audio(input_path, wav)

            # 3. Transcribir
            self._step("Transcribiendo con Whisper", 40)
            segments, meta = transcribe(wav, opts)

            if not segments:
                raise RuntimeError("Whisper no encontró habla en el audio.")

            # 4. Corrección LLM
            if opts.correct_with_llm:
                self._step("Corrigiendo con LLM", 65)
                segments = correct_segments(self._llm, segments,
                                            is_music=opts.use_demucs)

            # 5. Traducción LLM
            if opts.translate_to:
                self._step(f"Traduciendo a {opts.translate_to}", 80)
                segments = translate_segments(
                    self._llm, segments,
                    src_lang=meta["language"],
                    tgt_lang=opts.translate_to,
                )

            # 6. Formateo profesional
            self._step("Aplicando formato profesional", 92)
            segments = format_segments(segments)

        self._step("Listo", 100)
        return segments, meta
