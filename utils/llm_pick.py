import logging
import os
import warnings
from enum import Enum
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Filter framework warnings (e.g., fixed temperature sampling on flash-lite)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="langchain_google_genai",
)

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


MODEL_CONFIG: Dict[ModelTier, Dict[str, Any]] = {
    ModelTier.LOW: {
        "model": "gemini-3.5-flash-lite",
        # Omit temperature: fixed sampling defaults apply to this tier
    },
    ModelTier.MEDIUM: {
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
    },
    ModelTier.HIGH: {
        "model": "gemini-2.5-pro",
        "temperature": 0.0,
    },
}


def _resolve_api_key() -> str:
    """Resolves and returns a single API key, avoiding dual-key conflict warnings."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Neither 'GEMINI_API_KEY' nor 'GOOGLE_API_KEY' found in environment variables."
        )
    return api_key


def pick_llm(
    level: Union[str, ModelTier] = ModelTier.LOW,
    stop_after_attempt: int = 4,
):
    """Instantiates a ChatGoogleGenerativeAI model configured with LCEL rate-limit retries.

    Args:
        level: Performance tier ("low", "medium", or "high").
        stop_after_attempt: Maximum retry attempts for 429/quota exhaustion errors.

    Returns:
        A Runnable instance of ChatGoogleGenerativeAI wrapped with retry logic.
    """
    if isinstance(level, str):
        try:
            level = ModelTier(level.lower().strip())
        except ValueError:
            valid = [tier.value for tier in ModelTier]
            raise ValueError(f"Invalid tier '{level}'. Supported options: {valid}")

    config = MODEL_CONFIG[level].copy()
    api_key = _resolve_api_key()

    # max_retries=0 ensures LCEL's .with_retry() manages all retries without nested loops
    base_llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        max_retries=0,
        **config,
    )

    return base_llm.with_retry(
        retry_if_exception_type=(ResourceExhausted, TooManyRequests),
        stop_after_attempt=stop_after_attempt,
        wait_exponential_jitter=True,
    )


def extract_content(response: BaseMessage) -> str:
    """Safely extracts text string from ChatMessage response content,

    handling both plain strings and list-of-dicts formats.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    return str(content)


    