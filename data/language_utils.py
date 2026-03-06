"""
Language utilities for multilingual reasoning analysis.

Provides langdetect-based language detection and thinking-trace analysis
for Japanese, Bengali, and Swahili text.
"""

import re
from typing import Dict, List

import pandas as pd
from langdetect import LangDetectException, detect_langs
from langdetect.detector_factory import DetectorFactory

# Make langdetect deterministic across runs.
DetectorFactory.seed = 0

# Maps project language codes → ISO 639-1 codes used by langdetect.
_LANG_DETECT_CODE: Dict[str, str] = {
    "JA_JP": "ja",
    "BN_BD": "bn",
    "SW_KE": "sw",
}


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_language_ratio(text: str, target_language_code: str) -> Dict[str, float]:
    """Detect the language distribution of *text* using ``langdetect``.

    Calls ``langdetect.detect_langs`` on the full text to obtain a probability
    distribution over ISO 639-1 language codes, then maps those probabilities
    into three buckets:

    * **target** – probability mass assigned to the target language
      (``ja`` / ``bn`` / ``sw``).
    * **english** – probability mass assigned to ``en``.
    * **other** – the remaining probability (``1 − target − english``).

    ``DetectorFactory.seed = 0`` is set at import time to make results
    deterministic across runs.

    Args:
        text: The text to analyse.
        target_language_code: One of ``"JA_JP"``, ``"BN_BD"``, ``"SW_KE"``.

    Returns:
        ``{"target": float, "english": float, "other": float}`` where values
        sum to 1.0.  Returns ``{"target": 0.0, "english": 0.0, "other": 0.0}``
        for empty input, and ``{"target": 0.0, "english": 0.0, "other": 1.0}``
        when langdetect cannot identify any language.

    Raises:
        ValueError: If *target_language_code* is not supported.
    """
    if target_language_code not in _LANG_DETECT_CODE:
        raise ValueError(
            f"Unsupported language code: {target_language_code!r}. "
            f"Supported: {list(_LANG_DETECT_CODE)}."
        )

    if not text or not text.strip():
        return {"target": 0.0, "english": 0.0, "other": 0.0}

    target_lang = _LANG_DETECT_CODE[target_language_code]

    try:
        lang_probs = detect_langs(text)
    except LangDetectException:
        return {"target": 0.0, "english": 0.0, "other": 1.0}

    prob_map = {lp.lang: lp.prob for lp in lang_probs}
    target_prob = prob_map.get(target_lang, 0.0)
    english_prob = prob_map.get("en", 0.0)
    other_prob = max(0.0, 1.0 - target_prob - english_prob)

    return {
        "target": target_prob,
        "english": english_prob,
        "other": other_prob,
    }


# ---------------------------------------------------------------------------
# Thinking-trace analysis
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def analyze_thinking_language(response_text: str, language_code: str) -> Dict:
    """Extract the ``<think>`` block and analyse its language composition.

    Args:
        response_text: Full model response (may include ``<think>…</think>``).
        language_code: One of ``"JA_JP"``, ``"BN_BD"``, ``"SW_KE"``.

    Returns:
        A dict with keys ``target_ratio``, ``english_ratio``,
        ``other_ratio``, and ``thinking_length`` (character count of the
        extracted thinking trace).  If no ``<think>`` block is found, all
        ratios are ``0.0`` and ``thinking_length`` is ``0``.
    """
    match = _THINK_RE.search(response_text)
    if not match:
        return {
            "target_ratio": 0.0,
            "english_ratio": 0.0,
            "other_ratio": 0.0,
            "thinking_length": 0,
        }

    thinking_text = match.group(1)
    ratios = detect_language_ratio(thinking_text, language_code)

    return {
        "target_ratio": ratios["target"],
        "english_ratio": ratios["english"],
        "other_ratio": ratios["other"],
        "thinking_length": len(thinking_text),
    }


def batch_analyze_thinking_language(
    responses: List[Dict],
    language_code: str,
) -> pd.DataFrame:
    """Analyse thinking language for a batch of responses.

    Args:
        responses: A list of dicts, each containing at least a ``"response"``
            key whose value is the full model output string.
        language_code: One of ``"JA_JP"``, ``"BN_BD"``, ``"SW_KE"``.

    Returns:
        A :class:`~pandas.DataFrame` with columns ``target_ratio``,
        ``english_ratio``, ``other_ratio``, and ``thinking_length``.
        A final **summary** row is appended with column means.
    """
    rows = []
    for item in responses:
        analysis = analyze_thinking_language(item["response"], language_code)
        rows.append(analysis)

    df = pd.DataFrame(rows)

    # Append summary (mean) row
    summary = df.mean(numeric_only=True).to_dict()
    summary_df = pd.DataFrame([summary], index=["summary"])
    df = pd.concat([df, summary_df])

    return df
