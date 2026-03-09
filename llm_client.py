import os
from typing import Any, Optional

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL_NAME

_client: Optional[genai.Client] = None
_client_err: Optional[str] = None


def _get_gemini_client() -> genai.Client:
    global _client, _client_err

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key:
        _client_err = (
            "Gemini API key is not configured. Set GEMINI_API_KEY "
            "in your environment or in a local .env file."
        )
        raise RuntimeError(_client_err)

    try:
        _client = genai.Client(api_key=api_key)
        return _client
    except Exception as e:
        _client_err = str(e)
        raise RuntimeError(f"Failed to create Gemini client: {_client_err}") from e


def _resolve_model(task_type: str) -> str:
    _ = (task_type or "").strip().lower()
    return GEMINI_MODEL_NAME


def _build_generation_config(
    system_prompt: str,
    max_output_tokens: int,
    temperature: float,
    task_type: str,
) -> types.GenerateContentConfig:
    config_kwargs: dict[str, Any] = {
        "system_instruction": system_prompt,
        "max_output_tokens": max(1, int(max_output_tokens)),
        "temperature": float(temperature),
    }

    if task_type in {"quiz", "quiz_eval"}:
        config_kwargs["response_mime_type"] = "application/json"

    return types.GenerateContentConfig(**config_kwargs)


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(str(part_text))

    return "\n".join(parts).strip()


def call_llm(
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 2048,
    temperature: float = 0.3,
    task_type: str = "qa",
) -> str:
    """
    Call the configured Gemini model and return answer text.
    """
    client = _get_gemini_client()
    model = _resolve_model(task_type)

    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=_build_generation_config(
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            task_type=task_type,
        ),
    )

    text = _extract_response_text(resp)
    if text:
        return text

    raise RuntimeError("Gemini returned no text response.")
