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
        "english_system_prompt_mgsm": (
            "You are an expert math solver. Please reason step by step, and put your final answer within \\boxed{}."
        ),
        # Setting 2 — Native-language system prompt
        "native_system_prompt_mgsm": (
            "あなたは常に日本語で考える数学の専門家です。ステップバイステップで論理的に考えてください。最終的な答えは必ず \\boxed{} の中に書いてください。"
        ),
        "english_system_prompt_mmmlu": (
            'You are an expert in multiple-choice questions. Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'
        ),
        "native_system_prompt_mmmlu": (
            'あなたは常に日本語で考える多肢選択問題の専門家です。選択肢を日本語で論理的に考えて、最終的な答えを "answer": "C" の形式で出力してください。'
        ),
        # Thinking prefix injected into <think> for Setting 2
        "thinking_prefix": "ご依頼の通り、日本語で考え始めます。",
    },
    "BN_BD": {
        "language_name": "Bengali",
        "english_system_prompt_mgsm": (
            "You are an expert math solver. Please reason step by step, and put your final answer within \boxed{}."
        ),
        "native_system_prompt_mgsm": (
            "আপনি সবসময় বাংলায় চিন্তা করেন এমন একজন গণিত বিশেষজ্ঞ। ধাপে ধাপে যুক্তি দিয়ে সমাধান করুন এবং চূড়ান্ত উত্তর \boxed{} এর মধ্যে রাখুন।"
        ),
        "english_system_prompt_mmmlu": (
            'You are an expert in multiple-choice questions. Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'
        ),
        "native_system_prompt_mmmlu": (
            'আপনি সবসময় বাংলায় চিন্তা করেন এমন একজন বহু-নির্বাচনী প্রশ্নের বিশেষজ্ঞ। যুক্তি দিয়ে উত্তর দিন এবং চূড়ান্ত উত্তর "answer": "C" ফরম্যাটে দিন।'
        ),
        # Thinking prefix injected into <think> for Setting 2
        "thinking_prefix": "অনুরোধ অনুসারে, আমি বাংলায় চিন্তা শুরু করব।",
    },
    "SW_KE": {
        "language_name": "Swahili",
        "english_system_prompt_mgsm": (
            "You are an expert math solver. Please reason step by step, and put your final answer within \boxed{}."
        ),
        "native_system_prompt_mgsm": (
            "Wewe ni mtaalamu wa kutatua matatizo ya hisabati. Tafadhali fikiria hatua kwa hatua na uweke jibu lako la mwisho ndani ya \boxed{}."
        ),
        "english_system_prompt_mmmlu": (
            'You are an expert in multiple-choice questions. Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'
        ),
        "native_system_prompt_mmmlu": (
            'Wewe ni mtaalamu wa maswali ya chaguzi nyingi. Tafadhali onyesha chaguo lako katika sehemu ya jibu kwa herufi ya chaguo tu, kwa mfano, "answer": "C".'
        ),
        # Thinking prefix injected into <think> for Setting 2
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
