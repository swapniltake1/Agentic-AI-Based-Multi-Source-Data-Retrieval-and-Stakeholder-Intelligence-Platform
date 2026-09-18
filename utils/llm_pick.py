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


# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------
# Load environment variables from the .env file.
# The API key is never written to application logs.
load_dotenv()

# Use the module-level logger so messages can be identified easily
# in the centralized application log.
logger = logging.getLogger(__name__)

# Suppress the specific automatic function-calling warning generated
# by the Google integration.
warnings.filterwarnings(
    "ignore",
    message=".*Direct use of automatic function calling.*",
)


# ---------------------------------------------------------------------------
# Model Tiers
# ---------------------------------------------------------------------------
# Model tiers provide a simple way to select the required level of
# model capability throughout the application.
class ModelTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Gemini Model Configuration
# ---------------------------------------------------------------------------
# Centralized model configuration.
#
# Each tier defines:
# - Gemini model name
# - Reasoning effort
#
# Keeping this configuration in one place makes model changes easier.
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


# ---------------------------------------------------------------------------
# Fallback Order
# ---------------------------------------------------------------------------
# Defines which lower model tier should be used when the requested
# model is unavailable, rate-limited, or has exhausted its quota.
def _fallback_tiers(level: ModelTier) -> list[ModelTier]:
    """
    Return the requested model tier followed by its fallback tiers.

    Args:
        level: Requested model tier.

    Returns:
        List of unique model tiers in fallback order.
    """

    fallback_order = {
        ModelTier.LOW: [],
        ModelTier.MEDIUM: [ModelTier.LOW],
        ModelTier.HIGH: [ModelTier.MEDIUM, ModelTier.LOW],
    }

    tiers = [
        level,
        *fallback_order[level],
    ]

    # Avoid configuring the same model more than once if multiple
    # tiers happen to reference the same model.
    unique_tiers: list[ModelTier] = []
    seen_models: set[str] = set()

    for tier in tiers:
        model_name = MODEL_CONFIG[tier]["model"]

        if model_name not in seen_models:
            unique_tiers.append(tier)
            seen_models.add(model_name)

    logger.debug(
        "Resolved model fallback tiers: %s",
        [tier.value for tier in unique_tiers],
    )

    return unique_tiers


# ---------------------------------------------------------------------------
# API Key Resolution
# ---------------------------------------------------------------------------
def _resolve_api_key() -> str:
    """
    Resolve the Gemini API key from environment variables.

    GEMINI_API_KEY is preferred. GOOGLE_API_KEY is used as a fallback.

    Returns:
        str: Gemini API key.

    Raises:
        OSError: If no API key is configured.
    """

    gemini_key = os.getenv("GEMINI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    # Prefer the project-specific Gemini API key.
    if gemini_key:
        logger.debug("Using GEMINI_API_KEY")
        return gemini_key

    # Fall back to GOOGLE_API_KEY when GEMINI_API_KEY is unavailable.
    if google_key:
        logger.warning(
            "GEMINI_API_KEY not set. Falling back to GOOGLE_API_KEY."
        )
        return google_key

    # No credentials are available.
    logger.error("No Gemini API key found")

    raise OSError(
        "No Gemini API key found. "
        "Please set GEMINI_API_KEY in your .env file."
    )


# ---------------------------------------------------------------------------
# Base Model Builder
# ---------------------------------------------------------------------------
def get_base_llm(
    level: str | ModelTier = ModelTier.LOW,
) -> ChatGoogleGenerativeAI:
    """
    Create a base Gemini chat model for the requested tier.

    Args:
        level: Model tier as a string or ModelTier enum.

    Returns:
        Configured ChatGoogleGenerativeAI instance.

    Raises:
        ValueError: If an unsupported model tier is provided.
    """

    # Convert string input into the ModelTier enum.
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

            logger.error(
                "Invalid model tier requested: %s",
                level,
            )

            raise ValueError(
                f"Invalid tier '{level}'. Supported: {valid}"
            )

    config = MODEL_CONFIG[level].copy()

    # Resolve API credentials without exposing the key.
    api_key = _resolve_api_key()

    logger.debug(
        "Creating Gemini base model: %s",
        config["model"],
    )

    # Create and return the configured Gemini model.
    return ChatGoogleGenerativeAI(
        api_key=api_key,
        max_retries=3,
        **config,
    )


# ---------------------------------------------------------------------------
# Exceptions That Trigger Fallback
# ---------------------------------------------------------------------------
# These exceptions indicate that the current model may be unavailable,
# rate-limited, or otherwise unable to serve the request.
#
# In these cases, the configured lower-tier fallback model can be used.
QUOTA_AND_AVAILABILITY_EXCEPTIONS = (
    GoogleRateLimitError,
    GoogleModelNotFoundError,
    GoogleAPIError,
    ResourceExhausted,
    TooManyRequests,
    NotFound,
    ServiceUnavailable,
)


# ---------------------------------------------------------------------------
# Standard LLM Factory
# ---------------------------------------------------------------------------
def pick_llm(
    level: str | ModelTier = ModelTier.LOW,
    stop_after_attempt: int = 2,
    output_schema: Any | None = None,
) -> Runnable:
    """
    Create an LLM runnable with fallback and retry support.

    Args:
        level: Requested model tier.
        stop_after_attempt: Maximum number of retry attempts.
        output_schema: Optional schema for structured model output.

    Returns:
        Runnable configured with optional structured output,
        fallback models, and retry behavior.
    """

    # Normalize string-based model tier input.
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

            logger.error(
                "Invalid model tier requested: %s",
                level,
            )

            raise ValueError(
                f"Invalid tier '{level}'. Supported: {valid}"
            )

    # Determine requested model and fallback models.
    tiers = _fallback_tiers(level)

    logger.info(
        "Initializing LLM: %s",
        level.value,
    )

    runnables: list[Runnable] = []

    # Build a runnable for each configured tier.
    for tier in tiers:

        model_name = MODEL_CONFIG[tier]["model"]

        logger.debug(
            "Configuring Gemini model: %s",
            model_name,
        )

        base_llm = get_base_llm(tier)

        # Apply structured output when a schema is provided.
        if output_schema is not None:

            logger.debug(
                "Configuring structured output for model: %s",
                model_name,
            )

            runnable = base_llm.with_structured_output(
                output_schema
            )

        else:
            runnable = base_llm

        runnables.append(runnable)

    # The first runnable is always the requested model.
    runnable = runnables[0]

    # -----------------------------------------------------------------------
    # Configure Fallback Models
    # -----------------------------------------------------------------------
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

    else:
        logger.debug(
            "No fallback model configured for tier: %s",
            level.value,
        )

    # -----------------------------------------------------------------------
    # Configure Retry
    # -----------------------------------------------------------------------
    # Retry is used for temporary service availability/server errors.
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


# ---------------------------------------------------------------------------
# Tool LLM Factory
# ---------------------------------------------------------------------------
def get_tool_llm(
    level: str | ModelTier,
    tools: list[Any],
    stop_after_attempt: int = 2,
) -> Runnable:
    """
    Create a Gemini LLM configured for tool calling.

    Args:
        level: Requested model tier.
        tools: List of tools available to the model.
        stop_after_attempt: Maximum retry attempts.

    Returns:
        Runnable configured with tool calling, fallback models,
        and retry behavior.
    """

    # Normalize string-based model tier input.
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

            logger.error(
                "Invalid tool model tier requested: %s",
                level,
            )

            raise ValueError(
                f"Invalid tier '{level}'. Supported: {valid}"
            )

    # Resolve requested tier and fallback tiers.
    tiers = _fallback_tiers(level)

    logger.info(
        "Initializing tool LLM: %s",
        level.value,
    )

    logger.debug(
        "Tool LLM initialized with %d tool(s)",
        len(tools),
    )

    models: list[Runnable] = []

    # Create a tool-enabled runnable for each model tier.
    for tier in tiers:

        model_name = MODEL_CONFIG[tier]["model"]

        logger.debug(
            "Configuring tool model: %s",
            model_name,
        )

        base_llm = get_base_llm(tier)

        # Bind the supplied tools to the Gemini model.
        models.append(
            base_llm.bind_tools(tools)
        )

    # First model is the primary model.
    runnable = models[0]

    # -----------------------------------------------------------------------
    # Configure Fallback Models
    # -----------------------------------------------------------------------
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

    else:
        logger.debug(
            "No fallback model configured for tool tier: %s",
            level.value,
        )

    # -----------------------------------------------------------------------
    # Configure Retry
    # -----------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helper: Content Extraction
# ---------------------------------------------------------------------------
def extract_content(
    response: BaseMessage | str | list | Any,
) -> str:
    """
    Extract text content from different LLM response formats.

    Handles:
    - Plain strings
    - LangChain BaseMessage objects
    - Lists containing text dictionaries
    - Lists containing strings
    - Other response types

    Args:
        response: LLM response.

    Returns:
        str: Extracted textual content.
    """

    logger.debug(
        "Extracting content from response type: %s",
        type(response).__name__,
    )

    content = getattr(
        response,
        "content",
        response,
    )

    # Most common case: response content is already a string.
    if isinstance(content, str):
        return content

    # Gemini/LangChain may return structured content blocks.
    if isinstance(content, list):

        text_parts: list[str] = []

        for part in content:

            if isinstance(part, dict):

                text = part.get("text")

                if text:
                    text_parts.append(str(text))

            elif isinstance(part, str):

                text_parts.append(part)

        extracted_content = "".join(text_parts)

        logger.debug(
            "Extracted %d character(s) from structured response",
            len(extracted_content),
        )

        return extracted_content

    # Fallback for unexpected content types.
    logger.debug(
        "Converting unexpected response content type to string: %s",
        type(content).__name__,
    )

    return str(content)