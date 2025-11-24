# backend/ai_service.py - OpenAI Integration Service
import os
import traceback
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is niet gezet. "
        "Zet deze in .env (lokaal) of als Railway environment variable."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


def call_openai(prompt: str, max_tokens: int = 800) -> str:
    """
    Eenvoudige helper om een chat completion aan te roepen.
    Gooit een exception door bij fouten, zodat de endpoint dat kan afhandelen.
    """
    print("[OPENAI] Calling model:", OPENAI_MODEL)
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Je bent een behulpzame, neutrale Nederlandse tekstschrijver. "
                        "Geef geen juridisch, financieel of veiligheidsadvies."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=max_tokens,
        )
    except Exception as exc:
        print("[OPENAI][ERROR]", repr(exc))
        traceback.print_exc()
        raise

    content = completion.choices[0].message.content
    print("[OPENAI] Received completion (length", len(content or ""), ")")
    return content or ""
