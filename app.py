"""
app.py
======
Interfaz web Flask para generación de subtítulos .srt locales.

Rutas:
  GET  /                    → UI principal
  GET  /api/health          → estado de llama-server
  POST /api/jobs            → crear job (upload archivo o URL YouTube)
  GET  /api/jobs/<id>       → estado del job
  GET  /api/jobs/<id>/download → descargar .srt resultante
"""

import logging
import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from llm_client import LlamaServerClient
from pipeline import PipelineOptions, PipelineRunner, segments_to_srt
from youtube import download_video, is_youtube_url

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 ** 3  # 50 GB

JOBS_DIR = Path(__file__).parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

llm = LlamaServerClient(base_url="http://localhost:8080")

# jobs: {job_id: {status, step, percent, srt_path?, error?}}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _new_job() -> tuple[str, Path]:
    job_id  = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "step": "En cola", "percent": 0}
    return job_id, job_dir


def _fail(job_id: str, msg: str):
    log.error(f"[Job {job_id[:8]}] {msg}")
    with jobs_lock:
        jobs[job_id].update(status="error", step=msg, percent=0)


def _progress(job_id: str):
    def _cb(step: str, pct: int):
        with jobs_lock:
            jobs[job_id].update(status="running", step=step, percent=pct)
    return _cb


def _run_job(job_id: str, video_path: str, opts: PipelineOptions):
    try:
        runner = PipelineRunner(opts, llm, on_progress=_progress(job_id))
        segments, meta = runner.run(video_path)

        srt_content = segments_to_srt(segments)
        srt_name    = Path(video_path).stem + ".srt"
        srt_path    = JOBS_DIR / job_id / srt_name
        srt_path.write_text(srt_content, encoding="utf-8")

        with jobs_lock:
            jobs[job_id].update(
                status="done",
                step="Completado",
                percent=100,
                srt_path=str(srt_path),
                language=meta.get("language", "?"),
                duration=round(meta.get("duration", 0)),
                segments=len(segments),
            )
        log.info(f"[Job {job_id[:8]}] Completado. {len(segments)} segmentos.")

    except Exception as e:
        _fail(job_id, str(e))
    finally:
        # Limpiar archivo de video subido (no el .srt)
        vpath = Path(video_path)
        if vpath.exists() and vpath.suffix.lower() not in (".srt",):
            try:
                vpath.unlink()
            except Exception:
                pass


# ── Rutas ──────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html", llm_available=llm.is_available())


@app.get("/api/health")
def health():
    return jsonify({"llm_available": llm.is_available()})


@app.post("/api/jobs")
def create_job():
    form = request.form

    # Opciones del pipeline
    opts = PipelineOptions(
        language         = form.get("language", "auto"),
        whisper_model    = form.get("whisper_model", "medium"),
        use_demucs       = form.get("use_demucs") == "true",
        correct_with_llm = form.get("correct_with_llm") == "true",
        translate_to     = form.get("translate_to") or None,
    )

    job_id, job_dir = _new_job()

    # ── Caso 1: URL YouTube ───────────────────────────────────────────────────
    url = form.get("url", "").strip()
    if url:
        def run_yt():
            try:
                with jobs_lock:
                    jobs[job_id].update(status="running",
                                        step="Descargando video", percent=5)
                video_path = download_video(url, str(job_dir))
                _run_job(job_id, video_path, opts)
            except Exception as e:
                _fail(job_id, f"Error descargando YouTube: {e}")

        threading.Thread(target=run_yt, daemon=True).start()
        return jsonify({"job_id": job_id}), 202

    # ── Caso 2: archivo subido ────────────────────────────────────────────────
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "Debes subir un archivo o pegar una URL"}), 400

    safe_name  = Path(file.filename).name
    video_path = str(job_dir / safe_name)
    file.save(video_path)

    threading.Thread(target=_run_job, args=(job_id, video_path, opts),
                     daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job no encontrado"}), 404
    return jsonify({k: v for k, v in job.items() if k != "srt_path"})


@app.get("/api/jobs/<job_id>/download")
def job_download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "El job aún no está listo"}), 404

    srt_path = job.get("srt_path")
    if not srt_path or not Path(srt_path).exists():
        return jsonify({"error": "Archivo .srt no encontrado"}), 404

    return send_file(srt_path, as_attachment=True,
                     download_name=Path(srt_path).name, mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    log.info(f"VideoSubtítulos arrancando en http://localhost:{port}")
    log.info(f"llama-server: {'disponible' if llm.is_available() else 'no disponible (corrección LLM desactivada)'}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
