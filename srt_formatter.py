"""
srt_formatter.py
================
Formateador profesional de subtítulos según estándar Netflix/Warner.

Reglas aplicadas:
  CPL  : ≤42 caracteres por línea, máximo 2 líneas por bloque
  CPS  : ≤17 caracteres/segundo
  Dur  : entre 1 s y 7 s por subtítulo
  Gap  : ≥100 ms entre subtítulos consecutivos
  Líneas: segmentación en frontera sintáctica
  Pirámide: línea inferior más larga (o igual) que la superior
  Puntuación: sin punto final tras ? o !; guion + espacio en diálogos
"""

import re
import logging
from dataclasses import replace as dc_replace

log = logging.getLogger(__name__)

MAX_CPL = 42
MAX_CPS = 17
MIN_DUR = 1.0
MAX_DUR = 7.0
MIN_GAP = 0.100

_ARTICLES    = frozenset({"el","la","los","las","un","una","unos","unas","al","del"})
_PREPOSITIONS = frozenset({"a","ante","bajo","con","contra","de","desde","durante",
                            "en","entre","hacia","hasta","mediante","para","por",
                            "según","sin","sobre","tras","vía"})
_NEGATIONS   = frozenset({"no","ni","nunca","jamás","tampoco"})
_CLITICS     = frozenset({"me","te","se","le","les","lo","la","los","las","nos","os"})
_CONJUNCTIONS = frozenset({"y","e","o","u","pero","mas","sino","aunque","porque",
                            "que","cuando","como","donde","si","ni","ya","pues",
                            "mientras","entonces","luego","además","también"})


def _vlen(text: str) -> int:
    return len(re.sub(r'<[^>]+>', '', text))


def _find_split(words: list, max_chars: int) -> int:
    candidates = []
    acc = 0
    for i, w in enumerate(words):
        acc += len(w) + (1 if i > 0 else 0)
        if acc > max_chars:
            break
        if i < len(words) - 1:
            candidates.append(i + 1)

    if not candidates:
        return max(1, len(words) // 2)

    mid = len(words) / 2

    def score(idx: int) -> int:
        prev = words[idx - 1].lower().rstrip(",:;.!?—…\"')")
        nxt  = words[idx].lower().lstrip("¿¡\"'(")
        if prev in _ARTICLES:      return -100
        if prev in _PREPOSITIONS:  return -90
        if nxt in _NEGATIONS | _CLITICS: return -80
        pts = 0
        if words[idx - 1].endswith((",", ";", ":", "—")): pts += 50
        if nxt in _CONJUNCTIONS: pts += 30
        pts += max(0, 10 - int(abs(idx - mid) * 2))
        return pts

    return max(candidates, key=score)


def _split_lines(text: str, max_cpl: int = MAX_CPL) -> str:
    text = text.strip()
    if _vlen(text) <= max_cpl:
        return text

    words = text.split()
    split_at = _find_split(words, max_cpl)
    line1 = " ".join(words[:split_at])
    line2 = " ".join(words[split_at:])

    if _vlen(line1) > max_cpl or _vlen(line2) > max_cpl:
        mid = len(words) // 2
        line1, line2 = " ".join(words[:mid]), " ".join(words[mid:])

    # Pirámide: línea inferior ≥ superior
    w1 = line1.split()
    if len(w1) > 1:
        c1 = " ".join(w1[:-1])
        c2 = w1[-1] + " " + line2
        if _vlen(c2) <= max_cpl and _vlen(c2) >= _vlen(c1):
            line1, line2 = c1, c2

    return f"{line1}\n{line2}"


def _min_dur(text: str) -> float:
    return max(MIN_DUR, _vlen(text.replace("\n", " ")) / MAX_CPS)


def _punctuation(text: str) -> str:
    text = re.sub(r'([?!])\.$', r'\1', text)
    text = re.sub(r'([?!])\.\s+', r'\1 ', text)
    text = re.sub(r'^-(?!\s)', '- ', text, flags=re.MULTILINE)
    return text.strip()


def _split_long(seg, max_cpl: int) -> list:
    text = seg.text.strip()
    if _vlen(text) <= max_cpl * 2:
        return [seg]

    parts = re.split(r'(?<=[.!?…])\s+', text)
    if len(parts) < 2:
        parts = re.split(r',\s+', text, maxsplit=1)
    if len(parts) < 2:
        words = text.split()
        mid = len(words) // 2
        parts = [" ".join(words[:mid]), " ".join(words[mid:])]

    parts = [p.strip() for p in parts if p.strip()]
    total_chars = sum(_vlen(p) for p in parts) or 1
    total_dur   = seg.end - seg.start
    result, t   = [], seg.start

    for i, part in enumerate(parts):
        ratio = _vlen(part) / total_chars
        dur   = max(_min_dur(part), total_dur * ratio)
        t_end = min(seg.end, t + dur)
        result.append(dc_replace(seg, index=seg.index + i, start=t, end=t_end, text=part))
        t = t_end

    return result


def format_segments(segments: list) -> list:
    segs = [dc_replace(s, text=_punctuation(s.text)) for s in segments]

    expanded = []
    for s in segs:
        expanded.extend(_split_long(s, MAX_CPL))

    lined = [dc_replace(s, text=_split_lines(s.text)) for s in expanded]

    result = list(lined)
    for i, s in enumerate(result):
        min_needed  = max(MIN_DUR, _min_dur(s.text))
        max_allowed = s.start + MAX_DUR
        if i + 1 < len(result):
            max_allowed = min(max_allowed, result[i + 1].start - MIN_GAP)

        new_end = s.end
        if new_end - s.start < min_needed:
            new_end = s.start + min_needed
        if new_end > max_allowed:
            new_end = max_allowed
        if new_end < s.start + 0.1:
            new_end = s.start + 0.1
        result[i] = dc_replace(s, end=new_end)

    final = [dc_replace(s, index=i + 1) for i, s in enumerate(result)]
    log.info(f"[Formatter] {len(segments)}→{len(final)} segmentos")
    return final
