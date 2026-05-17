"""
youtube.py
==========
Descarga videos de YouTube (y otros sitios soportados por yt-dlp).
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def download_video(url: str, target_dir: str) -> str:
    """
    Descarga el video de la URL y lo guarda en target_dir.
    Devuelve la ruta completa al archivo descargado.
    """
    from yt_dlp import YoutubeDL

    out_tmpl = str(Path(target_dir) / "%(title).80s.%(ext)s")

    opts = {
        "format":           "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "outtmpl":          out_tmpl,
        "merge_output_format": "mp4",
        "quiet":            True,
        "no_warnings":      True,
        "noplaylist":       True,
    }

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    path = Path(filename)
    if not path.exists():
        # yt-dlp puede cambiar la extensión tras el merge
        for ext in ("mp4", "mkv", "webm", "avi"):
            alt = path.with_suffix(f".{ext}")
            if alt.exists():
                path = alt
                break

    log.info(f"[YouTube] Descargado: {path}")
    return str(path)


def is_youtube_url(url: str) -> bool:
    return any(d in url for d in ("youtube.com", "youtu.be", "yt.be"))
