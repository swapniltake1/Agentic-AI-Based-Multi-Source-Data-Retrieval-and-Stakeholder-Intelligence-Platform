import streamlit as st
from langchain_core.messages import HumanMessage

from agents.data_agent import data_agent


st.set_page_config(
    page_title="Data Intelligence Agent",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .app-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .app-subtitle {
        text-align: center;
        color: #777;
        margin-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="app-title"> Data Intelligence Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'Ask questions about your data, databases, APIs, or ETL operations.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.title("Data Agent")

    st.write(
        """
        This agent can route your request to:

        **SQL Analyst**
        - Query database data
        - Generate SQL
        - Analyze results

        **ETL Analyst**
        - Extract data
        - Transform data
        - Load transformed data
        """
    )

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# Display previous messages
# ---------------------------------------------------------

for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

user_input = st.chat_input("Message your data agent...")


if user_input:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # -----------------------------------------------------
    # Run agent
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = data_agent.invoke(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ],
                        "route_response": "",
                    }
                )

                # -------------------------------------------------
                # Extract final response
                # -------------------------------------------------

                messages = result.get("messages", [])

                assistant_text = None

                for msg in reversed(messages):

                    if hasattr(msg, "content") and msg.content:

                        assistant_text = msg.content
                        break

                if assistant_text is None:
                    assistant_text = "The agent completed the operation."

                st.markdown(assistant_text)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_text,
                    }
                )

            except Exception as e:

                error_message = f" Error: {str(e)}"

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )