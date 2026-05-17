#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "==> Creando entorno virtual..."
python3 -m venv venv

echo "==> Activando venv e instalando dependencias..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo ""
echo "==> Verificando dependencias del sistema..."
if command -v ffmpeg &>/dev/null; then
    echo "  ✓ ffmpeg $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3)"
else
    echo "  ✗ ffmpeg no encontrado. Instálalo con:"
    echo "      pkexec apt install ffmpeg"
fi

echo ""
echo "==> Verificando paquetes Python..."
python -c "import faster_whisper; print('  ✓ faster-whisper', faster_whisper.__version__)"
python -c "import flask; print('  ✓ Flask', flask.__version__)"
python -c "import yt_dlp; print('  ✓ yt-dlp', yt_dlp.version.__version__)"
python -c "import requests; print('  ✓ requests')"
python -c "import demucs; print('  ✓ demucs')" 2>/dev/null || echo "  ⚠ demucs no disponible (separación vocal desactivada)"

echo ""
echo "==> Verificando llama-server en localhost:8080..."
if curl -s http://localhost:8080/v1/models &>/dev/null; then
    echo "  ✓ llama-server activo"
else
    echo "  ⚠ llama-server no detectado (corrección LLM no disponible)"
    echo "    Inícialo con tu configuración habitual antes de usar la app."
fi

echo ""
echo "==> Setup completo. Para iniciar la app:"
echo "    ./run.sh"
echo "    Luego abre: http://localhost:5000"
