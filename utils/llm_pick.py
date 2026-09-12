import logging
import os
import warnings
from enum import Enum
from typing import Any

from dotenv import load_dotenv

from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    TooManyRequests,
)
from google.genai.errors import ServerError
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

load_dotenv()


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Suppress only the specific non-critical LangChain Google warning
# -------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message=".*Direct use of automatic function calling.*",
)


# -------------------------------------------------------------------
# Model Tiers
# -------------------------------------------------------------------

class ModelTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# -------------------------------------------------------------------
# Gemini Model Configuration
# -------------------------------------------------------------------

MODEL_CONFIG: dict[ModelTier, dict[str, Any]] = {

    # Lightweight model
    ModelTier.LOW: {
        "model": "gemini-3.5-flash-lite",
    },

    # General-purpose model
    ModelTier.MEDIUM: {
        "model": "gemini-3.6-flash",
    },

    # High tier
    ModelTier.HIGH: {
        "model": "gemini-3.6-flash",
    },
}


# -------------------------------------------------------------------
# API Key Resolution
# -------------------------------------------------------------------

def _resolve_api_key() -> str:
    """
    Resolve a single Gemini API key.

    GEMINI_API_KEY is preferred.
    GOOGLE_API_KEY is used only as a fallback.
    """

    gemini_key = os.getenv("GEMINI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if gemini_key:
        return gemini_key

    if google_key:
        logger.warning(
            "GEMINI_API_KEY is not set. "
            "Falling back to GOOGLE_API_KEY."
        )
        return google_key

    raise OSError(
        "No Gemini API key found.\n"
        "Please set GEMINI_API_KEY in your .env file."
    )


# -------------------------------------------------------------------
# LLM Factory
# -------------------------------------------------------------------

def pick_llm(
    level: str | ModelTier = ModelTier.LOW,
    stop_after_attempt: int = 4,
    output_schema: Any | None = None,
):
    """
    Create a Gemini LLM based on the requested model tier.

    IMPORTANT:
        Structured output MUST be applied before retry wrapping.

    Correct order:

        ChatGoogleGenerativeAI
                ↓
        with_structured_output()
                ↓
        with_retry()
                ↓
        RunnableRetry

    Args:
        level:
            Model tier:
                - low
                - medium
                - high

        stop_after_attempt:
            Maximum number of retry attempts for temporary failures.

        output_schema:
            Optional Pydantic schema for structured output.

    Returns:
        LangChain Runnable.
    """

    # ---------------------------------------------------------------
    # Convert string to ModelTier
    # ---------------------------------------------------------------

    if isinstance(level, str):
        try:
            level = ModelTier(level.lower().strip())

        except ValueError:
            valid = [tier.value for tier in ModelTier]

            raise ValueError(
                f"Invalid tier '{level}'. "
                f"Supported options: {valid}"
            )

    # ---------------------------------------------------------------
    # Get model configuration
    # ---------------------------------------------------------------

    config = MODEL_CONFIG[level].copy()

    # ---------------------------------------------------------------
    # Resolve API key
    # ---------------------------------------------------------------

    api_key = _resolve_api_key()

    # ---------------------------------------------------------------
    # Create base Gemini model
    # ---------------------------------------------------------------

    base_llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        max_retries=0,
        **config,
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # Structured output must be applied BEFORE with_retry().
    # ---------------------------------------------------------------

    if output_schema is not None:

        runnable = base_llm.with_structured_output(
            output_schema
        )

    else:

        runnable = base_llm

    # ---------------------------------------------------------------
    # Apply retry LAST
    # ---------------------------------------------------------------

    return runnable.with_retry(
        retry_if_exception_type=(
            ResourceExhausted,
            TooManyRequests,
            ServiceUnavailable,
            ServerError,
        ),
        stop_after_attempt=stop_after_attempt,
        wait_exponential_jitter=True,
    )


# -------------------------------------------------------------------
# Extract Response Content
# -------------------------------------------------------------------

def extract_content(
    response: BaseMessage | str | list | Any
) -> str:
    """
    Safely extract text from a LangChain response.

    Handles:
        - BaseMessage
        - Plain string
        - List of dictionaries
        - List of strings
        - Other response formats
    """

    # ---------------------------------------------------------------
    # Get content
    # ---------------------------------------------------------------

    content = getattr(response, "content", response)

    # ---------------------------------------------------------------
    # Normal string response
    # ---------------------------------------------------------------

    if isinstance(content, str):
        return content

    # ---------------------------------------------------------------
    # List-based response
    # ---------------------------------------------------------------

    if isinstance(content, list):

        text_parts: list[str] = []

        for part in content:

            if isinstance(part, dict):

                text = part.get("text")

                if text:
                    text_parts.append(str(text))

            elif isinstance(part, str):

                text_parts.append(part)

        return "".join(text_parts)

    # ---------------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------------

    return str(content)