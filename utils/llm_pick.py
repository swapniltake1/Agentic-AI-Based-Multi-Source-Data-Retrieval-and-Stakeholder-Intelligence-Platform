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
# Gemini Model Configuration (Free Tier Optimized)
# -------------------------------------------------------------------

MODEL_CONFIG: dict[ModelTier, dict[str, Any]] = {
    # Fast routing, simple checks, high daily free-tier volume
    ModelTier.LOW: {
        "model": "gemini-3.5-flash-lite",
        "reasoning_effort": "minimal",
    },
    # General data processing, SQL generation, and safety judge
    ModelTier.MEDIUM: {
        "model": "gemini-3.7-flash",
        "reasoning_effort": "low",  # 3.7-flash accepts 'low' or 'high' (rejects 'minimal')
    },
    # Complex agent reasoning, ETL orchestration, and multi-step tool calls
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
        return gemini_key

    if google_key:
        logger.warning("GEMINI_API_KEY not set. Falling back to GOOGLE_API_KEY.")
        return google_key

    raise OSError(
        "No Gemini API key found. Please set GEMINI_API_KEY in your .env file."
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
            raise ValueError(f"Invalid tier '{level}'. Supported: {valid}")

    config = MODEL_CONFIG[level].copy()
    api_key = _resolve_api_key()

    return ChatGoogleGenerativeAI(
        api_key=api_key,
        max_retries=3,  # Automatically retry 503 spikes before raising
        **config,
    )

# Exceptions that trigger instant fallback to the next tier
QUOTA_AND_AVAILABILITY_EXCEPTIONS = (
    GoogleRateLimitError,
    GoogleModelNotFoundError,
    GoogleAPIError,       # Handles 503 UNAVAILABLE and 500 server errors
    ResourceExhausted,
    TooManyRequests,
    NotFound,
    ServiceUnavailable,
)

# -------------------------------------------------------------------
# LLM Factories
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
            raise ValueError(f"Invalid tier '{level}'. Supported: {valid}")

    runnables: list[Runnable] = []
    tiers = _fallback_tiers(level)

    for tier in tiers:
        logger.info("Configuring Gemini model: %s", MODEL_CONFIG[tier]["model"])
        base_llm = get_base_llm(tier)

        if output_schema is not None:
            runnable = base_llm.with_structured_output(output_schema)
        else:
            runnable = base_llm

        runnables.append(runnable)

    runnable = runnables[0]

    if len(runnables) > 1:
        runnable = runnable.with_fallbacks(
            runnables[1:],
            exceptions_to_handle=QUOTA_AND_AVAILABILITY_EXCEPTIONS,
        )

    return runnable.with_retry(
        retry_if_exception_type=(ServiceUnavailable, ServerError),
        stop_after_attempt=stop_after_attempt,
        wait_exponential_jitter=True,
    )


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
            raise ValueError(f"Invalid tier '{level}'. Supported: {valid}")

    models: list[Runnable] = []
    tiers = _fallback_tiers(level)

    for tier in tiers:
        logger.info("Configuring tool model: %s", MODEL_CONFIG[tier]["model"])
        base_llm = get_base_llm(tier)
        models.append(base_llm.bind_tools(tools))

    runnable = models[0]

    if len(models) > 1:
        runnable = runnable.with_fallbacks(
            models[1:],
            exceptions_to_handle=QUOTA_AND_AVAILABILITY_EXCEPTIONS,
        )

    return runnable.with_retry(
        retry_if_exception_type=(ServiceUnavailable, ServerError),
        stop_after_attempt=stop_after_attempt,
        wait_exponential_jitter=True,
    )

# -------------------------------------------------------------------
# Helper: Content Extraction
# -------------------------------------------------------------------

def extract_content(response: BaseMessage | str | list | Any) -> str:
    content = getattr(response, "content", response)

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