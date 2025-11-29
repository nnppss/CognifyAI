import os
from typing import Optional

from google import genai

# Singleton client/cache
_client: Optional[genai.Client] = None
_client_err: Optional[str] = None


def _get_client() -> genai.Client:
    """
    Return a Gemini client. Requires GEMINI_API_KEY environment variable.
    """
    global _client, _client_err

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        _client_err = (
            "GEMINI_API_KEY is not set. Get a free key from Google AI Studio, "
            "then set it as an environment variable."
        )
        raise RuntimeError(_client_err)

    try:
        _client = genai.Client(api_key=api_key)
        return _client
    except Exception as e:
        _client_err = str(e)
        raise RuntimeError(f"Failed to create Gemini client: {_client_err}") from e


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the Gemini model with combined prompt and return answer text.
    """
    client = _get_client()

    prompt = f"{system_prompt}\n\n{user_prompt}"

    # Simple one-shot text generation call. :contentReference[oaicite:4]{index=4}
    resp = client.models.generate_content(
        model="gemini-2.5-flash",  # free-tier friendly model
        contents=prompt,
    )

    return (resp.text or "").strip()
