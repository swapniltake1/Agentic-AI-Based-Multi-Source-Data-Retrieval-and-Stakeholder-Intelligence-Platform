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
from langchain_core.runnables import Runnable
from langchain_google_genai.chat_models import GoogleRateLimitError
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
# Suppress specific non-critical LangChain Google warning
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


def _fallback_tiers(level: ModelTier) -> list[ModelTier]:
    fallback_order = {
        ModelTier.LOW: [],
        ModelTier.MEDIUM: [ModelTier.LOW],
        ModelTier.HIGH: [ModelTier.MEDIUM, ModelTier.LOW],
    }
    tiers = [level, *fallback_order[level]]
    unique_tiers = []
    seen_models = set()

    for tier in tiers:
        model_name = MODEL_CONFIG[tier]["model"]
        if model_name not in seen_models:
            unique_tiers.append(tier)
            seen_models.add(model_name)

    return unique_tiers


# -------------------------------------------------------------------
# Gemini Model Configuration
# -------------------------------------------------------------------

MODEL_CONFIG: dict[ModelTier, dict[str, Any]] = {

    ModelTier.LOW: {
        "model": "gemini-3.5-flash-lite",
        "model_kwargs": {
            "reasoning_efforts": "none",
        },
    },

    ModelTier.MEDIUM: {
        "model": "gemini-3.6-flash",
        "model_kwargs": {
            "reasoning_efforts": "none",
        },
    },

    ModelTier.HIGH: {
        "model": "gemini-3.6-flash",
        "model_kwargs": {
            "reasoning_efforts": "none",
        },
    },
}


# -------------------------------------------------------------------
# API Key Resolution
# -------------------------------------------------------------------

def _resolve_api_key() -> str:
    """
    Resolve Gemini API key.

    GEMINI_API_KEY is preferred.
    GOOGLE_API_KEY is used as fallback.
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
# Create Base Gemini LLM
# -------------------------------------------------------------------

def get_base_llm(
    level: str | ModelTier = ModelTier.LOW,
) -> ChatGoogleGenerativeAI:
    """
    Create and return the raw Gemini chat model.

    This function intentionally returns ChatGoogleGenerativeAI
    directly so methods such as bind_tools() are available to
    Pylance and at runtime.
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
    # Create Gemini model
    # ---------------------------------------------------------------

    return ChatGoogleGenerativeAI(
        api_key=api_key,
        max_retries=0,
        **config,
    )


# -------------------------------------------------------------------
# LLM Factory
# -------------------------------------------------------------------

def pick_llm(
    level: str | ModelTier = ModelTier.LOW,
    stop_after_attempt: int = 4,
    output_schema: Any | None = None,
) -> Runnable:
    """
    Create a Gemini LLM with optional structured output
    and retry handling.

    Processing order:

        ChatGoogleGenerativeAI
                ↓
        with_structured_output()
                ↓
            with_retry()
                ↓
          RunnableRetry
    """

    if isinstance(level, str):
        level = ModelTier(level.lower().strip())

    runnables = []

    for tier in _fallback_tiers(level):
        base_llm = get_base_llm(tier)

        if output_schema is not None:
            base_llm = base_llm.with_structured_output(output_schema)

        runnables.append(base_llm)

    runnable = runnables[0]

    if len(runnables) > 1:
        runnable = runnable.with_fallbacks(
            runnables[1:],
            exceptions_to_handle=(
                GoogleRateLimitError,
                ResourceExhausted,
                TooManyRequests,
            ),
        )

    # ---------------------------------------------------------------
    # Retry
    # ---------------------------------------------------------------

    return runnable.with_retry(
        retry_if_exception_type=(
            ServiceUnavailable,
            ServerError,
        ),
        stop_after_attempt=stop_after_attempt,
        wait_exponential_jitter=True,
    )


def get_tool_llm(level: str | ModelTier, tools: list[Any]):
    """Return a tool-bound LLM with quota fallbacks across model tiers."""

    if isinstance(level, str):
        level = ModelTier(level.lower().strip())

    models = [
        get_base_llm(tier).bind_tools(tools)
        for tier in _fallback_tiers(level)
    ]

    if len(models) == 1:
        return models[0]

    return models[0].with_fallbacks(
        models[1:],
        exceptions_to_handle=(
            GoogleRateLimitError,
            ResourceExhausted,
            TooManyRequests,
        ),
    )


# -------------------------------------------------------------------
# Extract Response Content
# -------------------------------------------------------------------

def extract_content(
    response: BaseMessage | str | list | Any
) -> str:
    """
    Safely extract text from a LangChain response.
    """

    # ---------------------------------------------------------------
    # Get content
    # ---------------------------------------------------------------

    content = getattr(response, "content", response)

    # ---------------------------------------------------------------
    # Normal string
    # ---------------------------------------------------------------

    if isinstance(content, str):
        return content

    # ---------------------------------------------------------------
    # List response
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