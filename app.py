import logging
import time

import streamlit as st

from logging_config import setup_logging


# ============================================================
# Logging Configuration
# ============================================================

# Initialize centralized application logging.
setup_logging()

logger = logging.getLogger(__name__)


# ============================================================
# LangChain / Agent Imports
# ============================================================

from langchain_core.messages import HumanMessage, AIMessage

from agents.data_agent import data_agent

from utils.llm_pick import (
    get_base_llm,
    MODEL_CONFIG,
    ModelTier,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Data Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
    }

    /* Chat input */
    div[data-testid="stChatInput"] {
        border-radius: 14px;
    }

    /* Status badges */
    .status-online {
        color: #16a34a;
        font-weight: 600;
    }

    .status-error {
        color: #dc2626;
        font-weight: 600;
    }

    .status-testing {
        color: #d97706;
        font-weight: 600;
    }

    .status-not-tested {
        color: #9ca3af;
        font-weight: 600;
    }

    /* Response metadata */
    .response-metadata {
        font-size: 12px;
        color: #9ca3af;
        margin-top: -5px;
        margin-bottom: 10px;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Streamlit Toolbar
# ============================================================

st.markdown(
    """
    <style>

    /* Keep Streamlit toolbar and three-dot menu visible */
    div[data-testid="stToolbar"] {
        visibility: visible;
    }

    /* Hide Deploy button */
    button[title="Deploy"] {
        display: none;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Constants
# ============================================================

# Maximum LangGraph recursion depth.
#
# This prevents an accidental graph loop from running forever.
AGENT_RECURSION_LIMIT = 20


# ============================================================
# Session State
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "model_status" not in st.session_state:
    st.session_state.model_status = {
        ModelTier.HIGH.value: "not_tested",
        ModelTier.MEDIUM.value: "not_tested",
        ModelTier.LOW.value: "not_tested",
    }


if "model_errors" not in st.session_state:
    st.session_state.model_errors = {}


if "run_model_tests" not in st.session_state:
    st.session_state.run_model_tests = False


# ============================================================
# Helper Functions
# ============================================================

def extract_message_text(message) -> str:
    """
    Extract text content from a LangChain message.

    Handles:
    - String content
    - List-based structured content
    - Dictionary text blocks
    """

    if not hasattr(message, "content"):
        return ""

    content = message.content

    # Normal text response.
    if isinstance(content, str):
        return content.strip()

    # Structured content response.
    if isinstance(content, list):

        text_parts = []

        for part in content:

            if isinstance(part, str):
                text_parts.append(part)

            elif isinstance(part, dict):

                text = part.get("text")

                if text:
                    text_parts.append(str(text))

        return "".join(text_parts).strip()

    if content is None:
        return ""

    return str(content).strip()


def extract_final_answer(result, route: str) -> str | None:
    """
    Extract the final user-facing AI response from the Data Agent.

    AIMessage is preferred so that:
    - HumanMessage is never shown as an answer.
    - Tool-call messages are not shown.
    - Intermediate agent messages are not accidentally displayed.

    Args:
        result: Data Agent result.
        route: SQL or ETL.

    Returns:
        Final AI response or None.
    """

    if not isinstance(result, dict):

        logger.error(
            "Data Agent returned unexpected result type: %s",
            type(result).__name__,
        )

        return None

    messages = result.get(
        "messages",
        [],
    )

    if not messages:

        logger.warning(
            "Data Agent returned no messages"
        )

        return None

    logger.debug(
        "Data Agent returned %d message(s)",
        len(messages),
    )

    # --------------------------------------------------------
    # Primary: AIMessage
    # --------------------------------------------------------

    for message in reversed(messages):

        if not isinstance(message, AIMessage):
            continue

        # Ignore AI messages that only contain tool calls.
        if getattr(
            message,
            "tool_calls",
            None,
        ):
            continue

        content = extract_message_text(message)

        if content:

            logger.info(
                "Final AI response extracted successfully"
            )

            return content

    # --------------------------------------------------------
    # Secondary fallback
    # --------------------------------------------------------
    #
    # This handles unusual child-agent responses where the
    # response may not be represented exactly as AIMessage.
    # HumanMessage is explicitly excluded.
    # --------------------------------------------------------

    for message in reversed(messages):

        if isinstance(message, HumanMessage):
            continue

        if getattr(
            message,
            "tool_calls",
            None,
        ):
            continue

        content = extract_message_text(message)

        if content:

            logger.warning(
                "Final response extracted from non-AIMessage: %s",
                type(message).__name__,
            )

            return content

    logger.warning(
        "No valid final AI response found for route: %s",
        route,
    )

    return None


def get_response_metadata(
    route: str,
    duration: float,
) -> str:
    """
    Build the small metadata line displayed below the AI response.

    Example:

        🗄️ SQL Analyst · Gemini Medium · 4.2s

    Args:
        route: Data Agent route.
        duration: Execution duration in seconds.

    Returns:
        Formatted metadata string.
    """

    # --------------------------------------------------------
    # Determine analyst
    # --------------------------------------------------------

    if route == "sql":

        analyst = "🗄️ SQL Analyst"

    elif route == "etl":

        analyst = "🔄 ETL Analyst"

    else:

        analyst = "🤖 Data Agent"

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    #
    # Current Data Agent configuration uses the medium tier.
    # Keep this display name user-friendly rather than exposing
    # the internal Gemini model identifier.
    # --------------------------------------------------------

    model_display = "Gemini Medium"

    # --------------------------------------------------------
    # Execution time
    # --------------------------------------------------------

    return (
        f"{analyst} · "
        f"{model_display} · "
        f"{duration:.1f}s"
    )


def get_safe_error_message(exception: Exception) -> str:
    """
    Convert internal exceptions into a safe frontend message.

    Detailed technical errors are logged but are not exposed
    directly to the user.
    """

    error_type = type(exception).__name__

    logger.error(
        "Agent request failed [%s]: %s",
        error_type,
        exception,
    )

    return (
        "I couldn't complete your request right now. "
        "Please try again."
    )


# ============================================================
# Model Test
# ============================================================

def test_model(tier: ModelTier) -> bool:
    """
    Test whether a Gemini model is responding.

    Returns:
        True when a response is received.
        False when the model fails or returns empty content.
    """

    model_name = MODEL_CONFIG[tier]["model"]

    logger.info(
        "Testing Gemini model: %s",
        model_name,
    )

    try:

        st.session_state.model_status[
            tier.value
        ] = "testing"

        llm = get_base_llm(tier)

        response = llm.invoke(
            "Reply with exactly: MODEL_TEST_OK"
        )

        content = getattr(
            response,
            "content",
            "",
        )

        if content:

            st.session_state.model_status[
                tier.value
            ] = "online"

            st.session_state.model_errors.pop(
                tier.value,
                None,
            )

            logger.info(
                "Model test successful: %s",
                model_name,
            )

            return True

        st.session_state.model_status[
            tier.value
        ] = "error"

        st.session_state.model_errors[
            tier.value
        ] = "Empty response"

        logger.warning(
            "Model returned an empty response: %s",
            model_name,
        )

        return False

    except Exception as e:

        st.session_state.model_status[
            tier.value
        ] = "error"

        st.session_state.model_errors[
            tier.value
        ] = str(e)

        logger.error(
            "Model test failed for %s: %s",
            model_name,
            e,
        )

        return False


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # Application Title
    # --------------------------------------------------------

    st.title("🤖 Data Agent")

    st.caption(
        "Multi-source data intelligence platform"
    )

    # --------------------------------------------------------
    # New Chat
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True,
    ):

        logger.info(
            "Starting new chat"
        )

        st.session_state.chat_history = []

        st.rerun()

    st.divider()

    # ========================================================
    # Model Status
    # ========================================================

    st.subheader("Model Status")

    model_order = [
        ModelTier.HIGH,
        ModelTier.MEDIUM,
        ModelTier.LOW,
    ]

    for tier in model_order:

        model_name = MODEL_CONFIG[tier]["model"]

        status = st.session_state.model_status[
            tier.value
        ]

        if status == "online":

            st.success(
                f"{model_name} — Online",
                icon="✅",
            )

        elif status == "error":

            st.error(
                f"{model_name} — Unavailable",
                icon="❌",
            )

        elif status == "testing":

            st.warning(
                f"{model_name} — Testing...",
                icon="🧪",
            )

        else:

            st.info(
                f"{model_name} — Not tested",
                icon="⚪",
            )

    # --------------------------------------------------------
    # Model Test Information
    # --------------------------------------------------------

    st.caption(
        "Running this test sends one real request "
        "to each model and consumes quota."
    )

    if st.button(
        "🧪 Test All Models",
        use_container_width=True,
    ):

        logger.info(
            "User requested model status test"
        )

        st.session_state.run_model_tests = True

        for tier in model_order:

            st.session_state.model_status[
                tier.value
            ] = "testing"

        st.rerun()

    # ========================================================
    # Execute Model Tests
    # ========================================================

    if st.session_state.run_model_tests:

        st.session_state.run_model_tests = False

        st.divider()

        st.write(
            "Running model tests..."
        )

        progress = st.progress(0)

        total_models = len(model_order)

        for index, tier in enumerate(model_order):

            model_name = MODEL_CONFIG[tier]["model"]

            with st.spinner(
                f"Testing {model_name}..."
            ):

                success = test_model(tier)

            if success:

                st.success(
                    f"{model_name} is online."
                )

            else:

                st.error(
                    f"{model_name} is unavailable."
                )

            progress.progress(
                (index + 1) / total_models
            )

        st.rerun()


# ============================================================
# Top Header
# ============================================================

header_col1, header_col2 = st.columns(
    [4, 1]
)

with header_col1:

    st.markdown(
        "### Data Intelligence Agent"
    )

    st.caption(
        "Multi-source data analysis • SQL • ETL"
    )

with header_col2:

    st.success(
        "● Ready"
    )


# ============================================================
# Welcome Screen
# ============================================================

if len(st.session_state.chat_history) == 0:

    st.markdown(
        "<h1 style='text-align:center;'>Hi Swapnil 👋</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; color:#6b7280;'>"
        "How can I help you with your data today?"
        "</p>",
        unsafe_allow_html=True,
    )

    st.write("")

    # --------------------------------------------------------
    # Capability Cards
    # --------------------------------------------------------

    card1, card2 = st.columns(
        2,
        gap="large",
    )

    with card1:

        with st.container(
            border=True,
        ):

            st.markdown(
                "### 🗄️ Data Analysis"
            )

            st.write(
                "Ask natural-language questions about "
                "your PostgreSQL database."
            )

            st.caption(
                "Generate SQL → validate → execute → answer"
            )

    with card2:

        with st.container(
            border=True,
        ):

            st.markdown(
                "### 🔄 ETL Operations"
            )

            st.write(
                "Extract data from APIs and transform "
                "files using Pandas."
            )

            st.caption(
                "Extract → Transform → Load"
            )

    st.write("")

    # --------------------------------------------------------
    # Suggested Questions
    # --------------------------------------------------------

    st.markdown(
        "#### Try something like"
    )

    suggestion1, suggestion2, suggestion3 = st.columns(
        3
    )

    suggestion_clicked = None

    with suggestion1:

        if st.button(
            "💳 Payment methods",
            use_container_width=True,
        ):

            suggestion_clicked = (
                "What are the different payment methods "
                "we have in our databases?"
            )

    with suggestion2:

        if st.button(
            "👥 Top 10 users",
            use_container_width=True,
        ):

            suggestion_clicked = (
                "Show me the top 10 users by number of rides."
            )

    with suggestion3:

        if st.button(
            "📊 Average ride distance",
            use_container_width=True,
        ):

            suggestion_clicked = (
                "What is the average ride distance?"
            )

    if suggestion_clicked:

        st.session_state.pending_question = (
            suggestion_clicked
        )

        st.rerun()


# ============================================================
# Chat History
# ============================================================

for chat in st.session_state.chat_history:

    if chat["role"] == "user":

        with st.chat_message(
            "user",
            avatar="👤",
        ):

            st.markdown(
                chat["content"]
            )

    else:

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            st.markdown(
                chat["content"]
            )


# ============================================================
# Chat Input
# ============================================================

user_input = st.chat_input(
    "Ask anything about your data..."
)


# ============================================================
# Suggested Question
# ============================================================

if "pending_question" in st.session_state:

    user_input = st.session_state.pending_question

    del st.session_state.pending_question


# ============================================================
# Process Question
# ============================================================

if user_input:

    user_input = user_input.strip()

    # --------------------------------------------------------
    # Validate User Input
    # --------------------------------------------------------

    if not user_input:

        logger.warning(
            "Empty user input received"
        )

        st.warning(
            "Please enter a question."
        )

        st.stop()

    logger.info(
        "Processing new user request"
    )

    # --------------------------------------------------------
    # Store User Question
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # --------------------------------------------------------
    # Display User Question
    # --------------------------------------------------------

    with st.chat_message(
        "user",
        avatar="👤",
    ):

        st.markdown(
            user_input
        )

    # --------------------------------------------------------
    # Assistant Response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖",
    ):

        try:

            # ==================================================
            # Execute Data Agent
            # ==================================================

            with st.spinner(
                "Analyzing your request..."
            ):

                start_time = time.perf_counter()

                logger.info(
                    "Calling Data Agent"
                )

                result = data_agent.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_input
                            )
                        ],
                        "route_response": "",
                    },
                    config={
                        "recursion_limit": AGENT_RECURSION_LIMIT,
                    },
                )

                duration = (
                    time.perf_counter()
                    - start_time
                )

                logger.info(
                    "Data Agent execution completed in %.2f seconds",
                    duration,
                )

            # ==================================================
            # Validate Result
            # ==================================================

            if not isinstance(result, dict):

                logger.error(
                    "Invalid Data Agent result type: %s",
                    type(result).__name__,
                )

                raise RuntimeError(
                    "Data Agent returned an invalid response."
                )

            # ==================================================
            # Get Route
            # ==================================================

            route = result.get(
                "route_response",
                "",
            )

            logger.info(
                "Data Agent route: %s",
                route if route else "unknown",
            )

            # ==================================================
            # Display Route
            # ==================================================

            if route == "sql":

                st.caption(
                    "🗄️ Routed to SQL Analyst"
                )

            elif route == "etl":

                st.caption(
                    "🔄 Routed to ETL Analyst"
                )

            # ==================================================
            # Extract Final AI Response
            # ==================================================

            final_answer = extract_final_answer(
                result,
                route,
            )

            # ==================================================
            # Handle Empty Response
            # ==================================================

            if not final_answer:

                logger.error(
                    "Data Agent completed without a usable final response"
                )

                st.error(
                    "The agent did not return a response. "
                    "Please try your question again."
                )

                # Do not save an empty/fake response.
                st.stop()

            # ==================================================
            # Display Actual AI Response
            # ==================================================

            st.markdown(
                final_answer
            )

            # ==================================================
            # Response Metadata
            # ==================================================

            response_metadata = get_response_metadata(
                route=route,
                duration=duration,
            )

            st.caption(
                response_metadata
            )

            # ==================================================
            # Save AI Response
            # ==================================================

            # Only the actual AI response is stored.
            # Metadata remains UI-only and is not saved.
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                }
            )

            logger.info(
                "AI response displayed and saved successfully"
            )

        # ======================================================
        # Error Handling
        # ======================================================

        except Exception as e:

            error_message = get_safe_error_message(
                e
            )

            st.error(
                error_message
            )

            # Store the safe user-facing error message.
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )