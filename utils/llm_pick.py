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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError


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


# -------------------------------------------------------------------
# Gemini Model Configuration
# -------------------------------------------------------------------

MODEL_CONFIG: dict[ModelTier, dict[str, Any]] = {

    # ---------------------------------------------------------------
    # LOW
    # ---------------------------------------------------------------

    ModelTier.LOW: {
        "model": "gemini-2.5-flash-lite",
        "model_kwargs": {
            "reasoning_efforts": "none",
        },
    },

    # ---------------------------------------------------------------
    # MEDIUM
    # ---------------------------------------------------------------

    ModelTier.MEDIUM: {
        "model": "gemini-2.5-flash",
        "model_kwargs": {
            "reasoning_efforts": "none",
        },
    },

    # ---------------------------------------------------------------
    # HIGH
    # ---------------------------------------------------------------

    ModelTier.HIGH: {
        "model": "gemini-2.5-flash",
        "model_kwargs": {
            "reasoning_efforts": "none",
        },
    },
}


# -------------------------------------------------------------------
# Fallback Order
# -------------------------------------------------------------------

def _fallback_tiers(level: ModelTier) -> list[ModelTier]:
    """
    Return model tiers in fallback order.

    Example:

        HIGH
          ↓
        MEDIUM
          ↓
        LOW
    """

    fallback_order = {

        ModelTier.LOW: [],

        ModelTier.MEDIUM: [
            ModelTier.LOW,
        ],

        ModelTier.HIGH: [
            ModelTier.MEDIUM,
            ModelTier.LOW,
        ],
    }

    tiers = [
        level,
        *fallback_order[level],
    ]

    unique_tiers: list[ModelTier] = []

    seen_models: set[str] = set()

    for tier in tiers:

        model_name = MODEL_CONFIG[tier]["model"]

        if model_name not in seen_models:

            unique_tiers.append(tier)

            seen_models.add(model_name)

    return unique_tiers


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
    Create the raw Gemini model.

    IMPORTANT:

    This function does NOT apply retry or fallback.

    This keeps bind_tools() and with_structured_output()
    available on the actual ChatGoogleGenerativeAI object.
    """

    # ---------------------------------------------------------------
    # Convert string to ModelTier
    # ---------------------------------------------------------------

    if isinstance(level, str):

        try:

            level = ModelTier(
                level.lower().strip()
            )

        except ValueError:

            valid = [
                tier.value
                for tier in ModelTier
            ]

            raise ValueError(
                f"Invalid tier '{level}'. "
                f"Supported options: {valid}"
            )

    # ---------------------------------------------------------------
    # Get configuration
    # ---------------------------------------------------------------

    config = MODEL_CONFIG[level].copy()

    # ---------------------------------------------------------------
    # Resolve API key
    # ---------------------------------------------------------------

    api_key = _resolve_api_key()

    # ---------------------------------------------------------------
    # Create model
    # ---------------------------------------------------------------

    return ChatGoogleGenerativeAI(
        api_key=api_key,
        max_retries=0,
        **config,
    )


# -------------------------------------------------------------------
# Exception Types For Quota Fallback
# -------------------------------------------------------------------

QUOTA_EXCEPTIONS = (
    GoogleRateLimitError,
    ResourceExhausted,
    TooManyRequests,
)


# -------------------------------------------------------------------
# LLM Factory
# -------------------------------------------------------------------

def pick_llm(
    level: str | ModelTier = ModelTier.LOW,
    stop_after_attempt: int = 2,
    output_schema: Any | None = None,
) -> Runnable:
    """
    Create an LLM with automatic model fallback.

    Example:

        HIGH
          ↓
        MEDIUM
          ↓
        LOW

    If a model returns a quota/rate-limit error,
    LangChain automatically attempts the next model.
    """

    # ---------------------------------------------------------------
    # Convert string
    # ---------------------------------------------------------------

    if isinstance(level, str):

        try:

            level = ModelTier(
                level.lower().strip()
            )

        except ValueError:

            valid = [
                tier.value
                for tier in ModelTier
            ]

            raise ValueError(
                f"Invalid tier '{level}'. "
                f"Supported options: {valid}"
            )

    # ---------------------------------------------------------------
    # Create model runnables
    # ---------------------------------------------------------------

    runnables: list[Runnable] = []

    tiers = _fallback_tiers(level)

    for tier in tiers:

        logger.info(
            "Configuring Gemini model: %s",
            MODEL_CONFIG[tier]["model"],
        )

        base_llm = get_base_llm(tier)

        # -----------------------------------------------------------
        # Structured output
        # -----------------------------------------------------------

        if output_schema is not None:

            runnable = base_llm.with_structured_output(
                output_schema
            )

        else:

            runnable = base_llm

        runnables.append(runnable)

    # ---------------------------------------------------------------
    # First model
    # ---------------------------------------------------------------

    runnable = runnables[0]

    # ---------------------------------------------------------------
    # Add quota fallbacks
    # ---------------------------------------------------------------

    if len(runnables) > 1:

        runnable = runnable.with_fallbacks(
            runnables[1:],
            exceptions_to_handle=QUOTA_EXCEPTIONS,
        )

    # ---------------------------------------------------------------
    # Retry only temporary server failures
    #
    # IMPORTANT:
    #
    # Do NOT retry quota errors here.
    # They should immediately move to the fallback model.
    # ---------------------------------------------------------------

    runnable = runnable.with_retry(
        retry_if_exception_type=(
            ServiceUnavailable,
            ServerError,
        ),
        stop_after_attempt=stop_after_attempt,
        wait_exponential_jitter=True,
    )

    return runnable


# -------------------------------------------------------------------
# Tool LLM
# -------------------------------------------------------------------

def get_tool_llm(
    level: str | ModelTier,
    tools: list[Any],
) -> Runnable:
    """
    Create a tool-enabled Gemini LLM with quota fallback.

    Example:

        HIGH model
             ↓
        quota exceeded
             ↓
        MEDIUM model
             ↓
        quota exceeded
             ↓
        LOW model
    """

    # ---------------------------------------------------------------
    # Convert string
    # ---------------------------------------------------------------

    if isinstance(level, str):

        try:

            level = ModelTier(
                level.lower().strip()
            )

        except ValueError:

            valid = [
                tier.value
                for tier in ModelTier
            ]

            raise ValueError(
                f"Invalid tier '{level}'. "
                f"Supported options: {valid}"
            )

    # ---------------------------------------------------------------
    # Create tool-bound models
    # ---------------------------------------------------------------

    models: list[Runnable] = []

    tiers = _fallback_tiers(level)

    for tier in tiers:

        logger.info(
            "Configuring tool model: %s",
            MODEL_CONFIG[tier]["model"],
        )

        base_llm = get_base_llm(tier)

        tool_llm = base_llm.bind_tools(
            tools
        )

        models.append(tool_llm)

    # ---------------------------------------------------------------
    # Single model
    # ---------------------------------------------------------------

    if len(models) == 1:

        return models[0]

    # ---------------------------------------------------------------
    # Model fallback
    # ---------------------------------------------------------------

    return models[0].with_fallbacks(
        models[1:],
        exceptions_to_handle=QUOTA_EXCEPTIONS,
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

        BaseMessage
        string
        list of dictionaries
        list of strings
        other response formats
    """

    # ---------------------------------------------------------------
    # Get content
    # ---------------------------------------------------------------

    content = getattr(
        response,
        "content",
        response,
    )

    # ---------------------------------------------------------------
    # String
    # ---------------------------------------------------------------

    if isinstance(content, str):

        return content

    # ---------------------------------------------------------------
    # List
    # ---------------------------------------------------------------

    if isinstance(content, list):

        text_parts: list[str] = []

        for part in content:

            if isinstance(part, dict):

                text = part.get("text")

                if text:

                    text_parts.append(
                        str(text)
                    )

            elif isinstance(part, str):

                text_parts.append(part)

        return "".join(text_parts)

    # ---------------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------------

    return str(content)