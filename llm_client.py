"""
llm_client.py
=============
Cliente HTTP al llama-server local (API OpenAI-compatible).

El llama-server corre en localhost:8080 con el modelo cargado
(actualmente Qwen3.5-9B-Q4_K_M). No gestiona el proceso del server:
si no está corriendo, is_available() devuelve False y el pipeline
continúa sin corrección LLM.
"""

import re
import logging
import requests
from dataclasses import replace as dc_replace

log = logging.getLogger(__name__)

LANG_NAMES = {
    "es": "español", "en": "inglés", "fr": "francés",
    "de": "alemán",  "it": "italiano", "pt": "portugués",
    "ja": "japonés", "zh": "chino",   "ru": "ruso",
    "ar": "árabe",   "ko": "coreano", "nl": "holandés",
    "pl": "polaco",  "sv": "sueco",   "tr": "turco",
}

_CORRECT_SYSTEM = (
    "Eres un editor profesional de subtítulos. Tu tarea es corregir errores "
    "evidentes de transcripción automática preservando EXACTAMENTE el formato dado. "
    "Aplica estas normas de subtitulado profesional:\n"
    "- Nunca pongas punto final después de ? o !\n"
    "- Usa guion + espacio (- texto) en líneas de diálogo simultáneo\n"
    "- Usa puntos suspensivos (...) para pausas, dudas o interrupciones\n"
    "- Si un subtítulo acaba interrumpido, el siguiente empieza con ...\n"
    "- Puedes condensar frases largas para facilitar la lectura\n"
    "- Escribe con letras los números del uno al nueve; cifras desde el 10\n"
    "Responde SOLO con el texto corregido, sin explicaciones ni comentarios."
)

_LYRICS_SYSTEM = (
    "Eres un editor de subtítulos de letras musicales. Corriges errores de "
    "transcripción automática en letras de canciones.\n"
    "REGLAS ESTRICTAS:\n"
    "- Preserva el estilo exacto de la letra: contracciones (gonna, ain't, 'cause), "
    "slang, repeticiones y cualquier forma no estándar son INTENCIONALES, no errores\n"
    "- Corrige SOLO palabras que no tienen ningún sentido en contexto musical "
    "(ruido, sílabas sueltas claramente erróneas)\n"
    "- Nunca 'corrijas' la gramática de la letra\n"
    "- Mantén el formato exacto: [NUMERO|INICIO-FIN] texto\n"
    "- Si una línea ya está bien, cópiala sin cambios\n"
    "Responde SOLO con las líneas, sin explicaciones."
)


class LlamaServerClient:
    def __init__(self, base_url: str = "http://localhost:8080", timeout: int = 120):
        self._base  = base_url.rstrip("/")
        self._timeout = timeout

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self._base}/v1/models", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, system: str, user: str,
             temperature: float = 0.1, max_tokens: int = 2048) -> str:
        payload = {
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      False,
        }
        r = requests.post(f"{self._base}/v1/chat/completions",
                          json=payload, timeout=self._timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def correct_segments(client: LlamaServerClient, segments: list,
                     batch_size: int = 20, is_music: bool = False) -> list:
    if not client.is_available():
        log.warning("[LLM] llama-server no disponible, omitiendo corrección")
        return segments

    system = _LYRICS_SYSTEM if is_music else _CORRECT_SYSTEM
    log.info(f"[LLM] Corrigiendo {len(segments)} segmentos "
             f"({'modo letras' if is_music else 'modo habla'})")
    corrected = list(segments)

    for i in range(0, len(corrected), batch_size):
        batch = corrected[i:i + batch_size]
        batch_text = "\n".join(
            f"[{s.index}|{s.start:.2f}-{s.end:.2f}] {s.text}" for s in batch
        )
        user_prompt = (
            "Corrige SOLO los errores evidentes (ortografía, puntuación, nombres propios).\n"
            "REGLAS:\n"
            "1. Mantén el formato: [NUMERO|INICIO-FIN] texto\n"
            "2. NO cambies el contenido ni el significado\n"
            "3. Si una línea está bien, cópiala igual\n"
            "4. Responde SOLO las líneas, sin texto adicional\n\n"
            f"TRANSCRIPCIÓN:\n{batch_text}"
        )
        try:
            response = client.chat(system, user_prompt, temperature=0.05)
            for line in response.strip().splitlines():
                m = re.match(r'\[(\d+)\|[\d.]+-[\d.]+\]\s*(.*)', line.strip())
                if m:
                    idx, text = int(m.group(1)), m.group(2).strip()
                    for j, seg in enumerate(corrected):
                        if seg.index == idx:
                            corrected[j] = dc_replace(seg, text=text)
                            break
            log.info(f"[LLM] Corrección lote {i // batch_size + 1} OK")
        except Exception as e:
            log.warning(f"[LLM] Error en corrección lote {i // batch_size + 1}: {e}")

    return corrected


def translate_segments(client: LlamaServerClient, segments: list,
                       src_lang: str, tgt_lang: str,
                       batch_size: int = 15) -> list:
    if not client.is_available():
        log.warning("[LLM] llama-server no disponible, omitiendo traducción")
        return segments

    src = LANG_NAMES.get(src_lang, src_lang)
    tgt = LANG_NAMES.get(tgt_lang, tgt_lang)
    log.info(f"[LLM] Traduciendo {src} → {tgt} ({len(segments)} segmentos)")

    system = (
        f"Eres un traductor profesional especializado en subtítulos. "
        f"Traduces de {src} a {tgt} con precisión y naturalidad. "
        f"Aplica estas normas de subtitulado profesional:\n"
        f"- Condensa y reduce frases largas cuando sea necesario\n"
        f"- Sin punto final después de ? o !\n"
        f"- Usa guion + espacio para diálogos simultáneos\n"
        f"- Usa ... para pausas e interrupciones\n"
        f"- Cada subtítulo debe poder leerse en el tiempo que dura (máx. 17 chars/s)\n"
        f"Responde SOLO con los subtítulos traducidos, sin explicaciones."
    )

    translated = list(segments)
    for i in range(0, len(translated), batch_size):
        batch = translated[i:i + batch_size]
        batch_text = "\n".join(f"[{s.index}] {s.text}" for s in batch)
        user_prompt = (
            f"Traduce al {tgt}.\n"
            "REGLAS ESTRICTAS:\n"
            "1. Formato: [NUMERO] texto traducido\n"
            "2. Traduce SOLO el texto, no los números\n"
            "3. Subtítulos naturales, fluidos y concisos\n"
            "4. Responde SOLO con los subtítulos\n\n"
            f"SUBTÍTULOS:\n{batch_text}"
        )
        try:
            response = client.chat(system, user_prompt, temperature=0.1)
            for line in response.strip().splitlines():
                m = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
                if m:
                    idx, text = int(m.group(1)), m.group(2).strip()
                    for j, seg in enumerate(translated):
                        if seg.index == idx:
                            translated[j] = dc_replace(seg, text=text)
                            break
            log.info(f"[LLM] Traducción lote {i // batch_size + 1} OK")
        except Exception as e:
            log.warning(f"[LLM] Error en traducción lote {i // batch_size + 1}: {e}")

    return translated
