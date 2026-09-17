import logging
import os
import warnings
from enum import Enum
from typing import Any

from dotenv import load_dotenv

from google.api_core.exceptions import (
    NotFound,
    ResourceExhausted,
    ServiceUnavailable,
    TooManyRequests,
)
from google.genai.errors import ServerError

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleModelNotFoundError,
    GoogleRateLimitError,
)


# -------------------------------------------------------------------
# Environment & Logging
# -------------------------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*Direct use of automatic function calling.*",
)


# -------------------------------------------------------------------
# Model Tiers & Free-Tier Configuration
# -------------------------------------------------------------------

class ModelTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# -------------------------------------------------------------------
# Gemini Model Configuration
# -------------------------------------------------------------------

MODEL_CONFIG: dict[ModelTier, dict[str, Any]] = {

    ModelTier.LOW: {
        "model": "gemini-3.5-flash-lite",
        "reasoning_effort": "minimal",
    },

    ModelTier.MEDIUM: {
        "model": "gemini-3.7-flash",
        "reasoning_effort": "low",
    },

    ModelTier.HIGH: {
        "model": "gemini-3.8-flash",
        "reasoning_effort": "medium",
    },
}


# -------------------------------------------------------------------
# Fallback Order
# -------------------------------------------------------------------

def _fallback_tiers(level: ModelTier) -> list[ModelTier]:

    fallback_order = {
        ModelTier.LOW: [],
        ModelTier.MEDIUM: [ModelTier.LOW],
        ModelTier.HIGH: [ModelTier.MEDIUM, ModelTier.LOW],
    }

    tiers = [level, *fallback_order[level]]

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

    gemini_key = os.getenv("GEMINI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if gemini_key:
        logger.debug("Using GEMINI_API_KEY")

        return gemini_key

    if google_key:

        logger.warning(
            "GEMINI_API_KEY not set. "
            "Falling back to GOOGLE_API_KEY."
        )

        return google_key

    logger.error(
        "No Gemini API key found."
    )

    raise OSError(
        "No Gemini API key found. "
        "Please set GEMINI_API_KEY in your .env file."
    )


# -------------------------------------------------------------------
# Base Model Builder
# -------------------------------------------------------------------

def get_base_llm(
    level: str | ModelTier = ModelTier.LOW,
) -> ChatGoogleGenerativeAI:

    if isinstance(level, str):

        try:
            level = ModelTier(level.lower().strip())

        except ValueError:

            valid = [tier.value for tier in ModelTier]

            logger.error(
                "Invalid model tier requested: %s",
                level,
            )

            raise ValueError(
                f"Invalid tier '{level}'. Supported: {valid}"
            )

    config = MODEL_CONFIG[level].copy()

    api_key = _resolve_api_key()

    logger.debug(
        "Creating Gemini model: %s",
        config["model"],
    )

    return ChatGoogleGenerativeAI(
        api_key=api_key,
        max_retries=3,
        **config,
    )


# -------------------------------------------------------------------
# Exceptions That Trigger Fallback
# -------------------------------------------------------------------

QUOTA_AND_AVAILABILITY_EXCEPTIONS = (
    GoogleRateLimitError,
    GoogleModelNotFoundError,
    GoogleAPIError,
    ResourceExhausted,
    TooManyRequests,
    NotFound,
    ServiceUnavailable,
)


# -------------------------------------------------------------------
# LLM Factory
# -------------------------------------------------------------------

def pick_llm(
    level: str | ModelTier = ModelTier.LOW,
    stop_after_attempt: int = 2,
    output_schema: Any | None = None,
) -> Runnable:

    if isinstance(level, str):

        try:
            level = ModelTier(level.lower().strip())

        except ValueError:

            valid = [tier.value for tier in ModelTier]

            logger.error(
                "Invalid model tier requested: %s",
                level,
            )

            raise ValueError(
                f"Invalid tier '{level}'. Supported: {valid}"
            )

    tiers = _fallback_tiers(level)

    logger.info(
        "Initializing LLM: %s",
        level.value,
    )

    runnables: list[Runnable] = []

    for tier in tiers:

        model_name = MODEL_CONFIG[tier]["model"]

        logger.debug(
            "Configuring Gemini model: %s",
            model_name,
        )

        base_llm = get_base_llm(tier)

        if output_schema is not None:

            runnable = base_llm.with_structured_output(
                output_schema
            )

        else:

            runnable = base_llm

        runnables.append(runnable)

    runnable = runnables[0]

    # ---------------------------------------------------------------
    # Configure fallback models
    # ---------------------------------------------------------------

    if len(runnables) > 1:

        fallback_models = [
            MODEL_CONFIG[tier]["model"]
            for tier in tiers[1:]
        ]

        logger.info(
            "LLM fallback configured: %s",
            fallback_models,
        )

        runnable = runnable.with_fallbacks(
            runnables[1:],
            exceptions_to_handle=QUOTA_AND_AVAILABILITY_EXCEPTIONS,
        )

    # ---------------------------------------------------------------
    # Configure retry
    # ---------------------------------------------------------------

    runnable = runnable.with_retry(
        retry_if_exception_type=(
            ServiceUnavailable,
            ServerError,
        ),
        stop_after_attempt=stop_after_attempt,
        wait_exponential_jitter=True,
    )

    logger.info(
        "LLM initialized successfully: %s",
        MODEL_CONFIG[level]["model"],
    )

    return runnable


# -------------------------------------------------------------------
# Tool LLM Factory
# -------------------------------------------------------------------

def get_tool_llm(
    level: str | ModelTier,
    tools: list[Any],
    stop_after_attempt: int = 2,
) -> Runnable:

    if isinstance(level, str):

        try:
            level = ModelTier(level.lower().strip())

        except ValueError:

            valid = [tier.value for tier in ModelTier]

            logger.error(
                "Invalid tool model tier requested: %s",
                level,
            )

            raise ValueError(
                f"Invalid tier '{level}'. Supported: {valid}"
            )

    tiers = _fallback_tiers(level)

    logger.info(
        "Initializing tool LLM: %s",
        level.value,
    )

    models: list[Runnable] = []

    for tier in tiers:

        model_name = MODEL_CONFIG[tier]["model"]

        logger.debug(
            "Configuring tool model: %s",
            model_name,
        )

        base_llm = get_base_llm(tier)

        models.append(
            base_llm.bind_tools(tools)
        )

    runnable = models[0]

    # ---------------------------------------------------------------
    # Configure fallback models
    # ---------------------------------------------------------------

    if len(models) > 1:

        fallback_models = [
            MODEL_CONFIG[tier]["model"]
            for tier in tiers[1:]
        ]

        logger.info(
            "Tool LLM fallback configured: %s",
            fallback_models,
        )

        runnable = runnable.with_fallbacks(
            models[1:],
            exceptions_to_handle=QUOTA_AND_AVAILABILITY_EXCEPTIONS,
        )

    # ---------------------------------------------------------------
    # Configure retry
    # ---------------------------------------------------------------

    runnable = runnable.with_retry(
        retry_if_exception_type=(
            ServiceUnavailable,
            ServerError,
        ),
        stop_after_attempt=stop_after_attempt,
        wait_exponential_jitter=True,
    )

    logger.info(
        "Tool LLM initialized successfully: %s",
        MODEL_CONFIG[level]["model"],
    )

    return runnable


# -------------------------------------------------------------------
# Helper: Content Extraction
# -------------------------------------------------------------------

def extract_content(
    response: BaseMessage | str | list | Any,
) -> str:

    content = getattr(
        response,
        "content",
        response,
    )

    if isinstance(content, str):

        return content

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

    return str(content)