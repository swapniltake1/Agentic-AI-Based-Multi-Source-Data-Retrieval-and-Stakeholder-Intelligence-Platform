import streamlit as st
from langchain_core.messages import HumanMessage

from agents.data_agent import data_agent
from utils.llm_pick import get_base_llm, MODEL_CONFIG, ModelTier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Data Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
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
# SESSION STATE
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
# MODEL TEST
# ============================================================

def test_model(tier: ModelTier):

    model_name = MODEL_CONFIG[tier]["model"]

    try:

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

            return True

        st.session_state.model_status[
            tier.value
        ] = "error"

        st.session_state.model_errors[
            tier.value
        ] = "Empty response"

        return False

    except Exception as e:

        st.session_state.model_status[
            tier.value
        ] = "error"

        st.session_state.model_errors[
            tier.value
        ] = str(e)

        return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # Application title
    # --------------------------------------------------------

    st.title("🤖 Data Agent")

    st.caption(
        "Multi-source data intelligence platform"
    )


    # --------------------------------------------------------
    # New chat
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True,
    ):

        st.session_state.chat_history = []

        st.rerun()


    st.divider()


    # --------------------------------------------------------
    # Capabilities
    # --------------------------------------------------------

    st.subheader("Capabilities")

    st.markdown(
        """
        **🗄️ SQL Analysis**

        Ask natural-language questions about your database.

        **🔄 ETL Operations**

        Extract, transform and load data from APIs and files.

        **🧠 Intelligent Routing**

        Automatically routes requests to the appropriate agent.
        """
    )


    st.divider()


 # ============================================================
# MODEL STATUS
# ============================================================

    st.subheader("Model Status")

    st.caption(
        "Gemini models configured in your application"
    )

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
    # Test button
    # --------------------------------------------------------

    st.caption(
        "Running this test sends one real request "
        "to each model and consumes quota."
    )


    if st.button(
        "🧪 Test All Models",
        use_container_width=True,
    ):

        st.session_state.run_model_tests = True

        for tier in model_order:

            st.session_state.model_status[
                tier.value
            ] = "testing"

        st.rerun()


    # ========================================================
    # EXECUTE MODEL TESTS
    # ========================================================

    if st.session_state.run_model_tests:

        st.session_state.run_model_tests = False

        st.divider()

        st.write("Running model tests...")


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


    st.divider()


    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    st.subheader("Example Questions")

    st.markdown(
        """
        **SQL**

        • What payment methods do we have?

        • Show the top 10 users.

        • What is the average ride distance?

        **ETL**

        • Extract data from an API.

        • Save API data as CSV.

        • Transform my CSV data.
        """
    )


    st.divider()

    st.caption(
        "Data Intelligence Platform"
    )


# ============================================================
# TOP HEADER
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
        "● Ready",
        icon="🤖",
    )


st.divider()


# ============================================================
# WELCOME SCREEN
# ============================================================

if len(st.session_state.chat_history) == 0:

    # --------------------------------------------------------
    # Welcome
    # --------------------------------------------------------

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
    # Capability cards
    # --------------------------------------------------------

    card1, card2 = st.columns(
        2,
        gap="large",
    )


    with card1:

        with st.container(
            border=True,
        ):

            st.markdown("### 🗄️ SQL Analysis")

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

            st.markdown("### 🔄 ETL Operations")

            st.write(
                "Extract data from APIs and transform "
                "files using Pandas."
            )

            st.caption(
                "Extract → Transform → Load"
            )


    st.write("")


    # --------------------------------------------------------
    # Suggested questions
    # --------------------------------------------------------

    st.markdown("#### Try something like")


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
# CHAT HISTORY
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
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask anything about your data..."
)


# ============================================================
# SUGGESTED QUESTION
# ============================================================

if "pending_question" in st.session_state:

    user_input = st.session_state.pending_question

    del st.session_state.pending_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_input:

    # --------------------------------------------------------
    # Store user question
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )


    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message(
        "user",
        avatar="👤",
    ):

        st.markdown(
            user_input
        )


    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖",
    ):

        try:

            with st.spinner(
                "Analyzing your request..."
            ):

                # ==========================================
                # CALL DATA AGENT
                # ==========================================

                result = data_agent.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_input
                            )
                        ],
                        "route_response": "",
                    }
                )


                # ==========================================
                # GET ROUTE
                # ==========================================

                route = result.get(
                    "route_response",
                    "",
                )


                # ==========================================
                # DISPLAY ROUTE
                # ==========================================

                if route:

                    if route == "sql":

                        st.caption(
                            "🗄️ Routed to SQL Analyst"
                        )

                    elif route == "etl":

                        st.caption(
                            "🔄 Routed to ETL Analyst"
                        )


                # ==========================================
                # FIND FINAL RESPONSE
                # ==========================================

                final_answer = None

                messages = result.get(
                    "messages",
                    [],
                )


                for message in reversed(
                    messages
                ):

                    if not hasattr(
                        message,
                        "content",
                    ):

                        continue


                    content = message.content


                    if not content:

                        continue


                    if getattr(
                        message,
                        "tool_calls",
                        None,
                    ):

                        continue


                    final_answer = content

                    break


                # ==========================================
                # FALLBACK
                # ==========================================

                if not final_answer:

                    final_answer = (
                        "The agent completed "
                        "the requested operation."
                    )


                # ==========================================
                # SHOW ANSWER
                # ==========================================

                st.markdown(
                    final_answer
                )


                # ==========================================
                # SAVE
                # ==========================================

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": final_answer,
                    }
                )


        except Exception as e:

            error_message = (
                "Something went wrong while "
                "processing your request.\n\n"
                f"`{str(e)}`"
            )

            st.error(
                error_message
            )


            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )