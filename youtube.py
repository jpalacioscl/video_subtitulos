"""
youtube.py
==========
Descarga videos y subtítulos de YouTube (y otros sitios soportados por yt-dlp).
"""

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


def download_video(url: str, target_dir: str) -> str:
    """Descarga el video y devuelve la ruta completa al archivo."""
    from yt_dlp import YoutubeDL

    out_tmpl = str(Path(target_dir) / "%(title).80s.%(ext)s")

    opts = {
        "format":              "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "outtmpl":             out_tmpl,
        "merge_output_format": "mp4",
        "quiet":               True,
        "no_warnings":         True,
        "noplaylist":          True,
    }

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    path = Path(filename)
    if not path.exists():
        for ext in ("mp4", "mkv", "webm", "avi"):
            alt = path.with_suffix(f".{ext}")
            if alt.exists():
                path = alt
                break

    log.info(f"[YouTube] Descargado: {path}")
    return str(path)


def fetch_subtitles(url: str, lang: str = "en") -> list | None:
    """
    Intenta obtener subtítulos de YouTube (manuales primero, luego auto-generados).
    Devuelve lista de Segment-like dicts {start, end, text} o None si no hay.
    """
    from yt_dlp import YoutubeDL

    opts = {
        "quiet":             True,
        "no_warnings":       True,
        "noplaylist":        True,
        "skip_download":     True,
        "writesubtitles":    True,
        "writeautomaticsub": True,
        "subtitleslangs":    [lang, f"{lang}-orig"],
        "subtitlesformat":   "vtt",
    }

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

        subs = info.get("subtitles", {})
        auto = info.get("automatic_captions", {})

        # Intentar subtítulos manuales primero, luego automáticos
        for source, label in [(subs, "manual"), (auto, "auto")]:
            for key in (lang, f"{lang}-orig", lang[:2]):
                entries = source.get(key)
                if entries:
                    vtt_url = next((e["url"] for e in entries
                                    if e.get("ext") == "vtt"), None)
                    if vtt_url:
                        segments = _parse_vtt_url(vtt_url)
                        if segments:
                            log.info(f"[YouTube] Subtítulos {label} ({key}): "
                                     f"{len(segments)} segmentos")
                            return segments
    except Exception as e:
        log.warning(f"[YouTube] No se pudieron obtener subtítulos: {e}")

    return None


def _parse_vtt_url(url: str) -> list | None:
    """Descarga y parsea un VTT desde una URL."""
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            text = r.read().decode("utf-8", errors="replace")
        return _parse_vtt(text)
    except Exception as e:
        log.warning(f"[YouTube] Error descargando VTT: {e}")
        return None


_VTT_TS = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
)
_HTML_TAG = re.compile(r"<[^>]+>")


def _ts_to_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_vtt(text: str) -> list:
    segments = []
    idx = 1
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _VTT_TS.match(lines[i].strip())
        if m:
            start = _ts_to_sec(*m.group(1, 2, 3, 4))
            end   = _ts_to_sec(*m.group(5, 6, 7, 8))
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                clean = _HTML_TAG.sub("", lines[i]).strip()
                if clean:
                    text_lines.append(clean)
                i += 1
            text = " ".join(text_lines)
            # deduplicar líneas repetidas del VTT de YouTube
            if text and (not segments or segments[-1]["text"] != text):
                segments.append({"index": idx, "start": start,
                                 "end": end, "text": text})
                idx += 1
        else:
            i += 1
    return segments


def is_youtube_url(url: str) -> bool:
    return any(d in url for d in ("youtube.com", "youtu.be", "yt.be"))
