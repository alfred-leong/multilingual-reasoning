"""
Prompts for the Translate-DPO experiment.

Setting 1 (English response): Standard English system prompt; the model
    reasons in English.
Setting 2 (Native response): Native-language system prompt + a thinking
    prefix that seeds the ``<think>`` block in the target language.
"""

from typing import Dict

# ---------------------------------------------------------------------------
# Per-language prompt definitions
# ---------------------------------------------------------------------------

_TRANSLATE_DPO_PROMPTS: Dict[str, Dict[str, str]] = {
    "JA_JP": {
        "language_name": "Japanese",
        # Setting 1 — English system prompt
        "english_system_prompt": (
            "You are a helpful assistant. "
            "Solve the problem step-by-step."
        ),
        # Setting 2 — Native-language system prompt
        "native_system_prompt": (
            "あなたは常に日本語で考える親切なアシスタントです。"
            "問題をステップバイステップで解いてください。"
        ),
        # Thinking prefix injected into <think> for Setting 2
        "thinking_prefix": "ご依頼の通り、日本語で考え始めます。",
    },
    "BN_BD": {
        "language_name": "Bengali",
        "english_system_prompt": (
            "You are a helpful assistant. "
            "Solve the problem step-by-step."
        ),
        "native_system_prompt": (
            "আপনি একজন সহায়ক সহকারী যিনি সবসময় বাংলায় চিন্তা করেন। "
            "সমস্যাটি ধাপে ধাপে সমাধান করুন।"
        ),
        "thinking_prefix": "অনুরোধ অনুসারে, আমি বাংলায় চিন্তা শুরু করব।",
    },
    "SW_KE": {
        "language_name": "Swahili",
        "english_system_prompt": (
            "You are a helpful assistant. "
            "Solve the problem step-by-step."
        ),
        "native_system_prompt": (
            "Wewe ni msaidizi wa manufaa ambaye hufikiri kila wakati kwa Kiswahili. "
            "Tatua tatizo hatua kwa hatua."
        ),
        "thinking_prefix": "Kwa ombi, nitaanza kufikiria kwa Kiswahili.",
    },
}


def get_translate_dpo_prompts(language_code: str) -> Dict[str, str]:
    """Return prompts for the Translate-DPO experiment.

    Args:
        language_code: One of ``"JA_JP"``, ``"BN_BD"``, ``"SW_KE"``.

    Returns:
        Dict with keys ``language_name``, ``english_system_prompt``,
        ``native_system_prompt``, ``thinking_prefix``.

    Raises:
        ValueError: If *language_code* is not supported.
    """
    if language_code not in _TRANSLATE_DPO_PROMPTS:
        raise ValueError(
            f"Unsupported language code: {language_code!r}. "
            f"Supported: {list(_TRANSLATE_DPO_PROMPTS)}"
        )
    return _TRANSLATE_DPO_PROMPTS[language_code]
