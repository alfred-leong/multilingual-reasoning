#!/usr/bin/env python3
"""
Custom vLLM launcher that patches a rope_parameters bug for Gemma3 models.

vLLM 0.16.0 has a bug in its Transformers v4 backward-compatibility path:
when a model config has both ``rope_theta`` (scalar) and a **nested**
``rope_parameters`` dict (keyed by attention layer type, e.g. Gemma3), vLLM
injects ``rope_theta`` as a flat key into the nested dict.  This breaks
``is_rope_parameters_nested()`` — which checks that *every* key is an
attention-layer-type name — and the server crashes with::

    rope_parameters should have a 'rope_type' key

This script monkey-patches ``patch_rope_parameters`` to skip the faulty
injection when ``rope_parameters`` is nested, then delegates to the
standard ``vllm serve`` entry-point.

Usage (drop-in replacement for ``vllm serve``):

    python scripts/launch_vllm_patched.py google/translategemma-4b-it \\
        --host 127.0.0.1 --port 8400 --dtype bfloat16 ...
"""

import sys


def _apply_monkey_patch():
    """Patch ``vllm.transformers_utils.config.patch_rope_parameters``."""
    import vllm.transformers_utils.config as vtc

    _original_patch_rope_parameters = vtc.patch_rope_parameters

    def _patched_patch_rope_parameters(config):
        """Fixed version that skips scalar injection into nested dicts."""
        rope_params = getattr(config, "rope_parameters", None)
        if rope_params and isinstance(rope_params, dict):
            if vtc.is_rope_parameters_nested(rope_params):
                # Nested rope_parameters (e.g. Gemma3): call
                # patch_rope_parameters_dict on each entry individually
                # WITHOUT injecting scalar fields into the parent dict.
                for entry in rope_params.values():
                    vtc.patch_rope_parameters_dict(entry)
                return

        # Fall back to original for non-nested / no rope_parameters cases
        _original_patch_rope_parameters(config)

    vtc.patch_rope_parameters = _patched_patch_rope_parameters


def _disable_chat_warmup():
    """Disable chat-template warmup for TranslateGemma.

    TranslateGemma's Jinja chat template expects non-standard content-part
    fields (``source_lang_code``, ``target_lang_code``) that vLLM's OpenAI
    layer strips.  Since we use the raw completions API instead of chat,
    the warmup is unnecessary and would log a confusing error.
    """
    import vllm.entrypoints.openai.chat_completion.serving as ccs

    _OrigChatServing = ccs.OpenAIServingChat

    class _PatchedChatServing(_OrigChatServing):
        async def warmup(self) -> None:  # noqa: D102
            pass  # skip warmup entirely for TranslateGemma

    ccs.OpenAIServingChat = _PatchedChatServing


def main():
    # Apply the fix before vLLM imports touch the config machinery
    _apply_monkey_patch()
    _disable_chat_warmup()

    # Delegate to the real vLLM CLI (same as running `vllm serve ...`)
    from vllm.entrypoints.cli.main import main as vllm_main

    # vllm CLI expects sys.argv[0] to be "vllm" and sys.argv[1] to be "serve"
    # Rewrite argv so the serve sub-command is selected.
    sys.argv = ["vllm", "serve"] + sys.argv[1:]
    vllm_main()


if __name__ == "__main__":
    main()
