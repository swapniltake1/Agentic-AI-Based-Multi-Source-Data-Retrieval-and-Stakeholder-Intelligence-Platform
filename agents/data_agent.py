import os
import sys
import logging

# Allow imports when running the project from the root directory
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from models.schema import RouterSchema, DataAgentSchema
from utils.llm_pick import get_base_llm

from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst

from logging_config import setup_logging


# ============================================================
# Logging Configuration
# ============================================================

setup_logging()

logger = logging.getLogger(__name__)


# ============================================================
# Parent Agent
# ============================================================

logger.info("Initializing Data Agent")

llm = get_base_llm("medium")

agent_router = llm.with_structured_output(RouterSchema)

logger.info("Data Agent router initialized")


# ============================================================
# Router Node
# ============================================================

def router_node(state: DataAgentSchema):

    logger.info("Router started")

    try:

        message = state.messages[-1].content

        route_response = agent_router.invoke(message)

        state.route_response = route_response.answer

        logger.info(
            "Request routed to: %s",
            state.route_response
        )

        return state

    except Exception as e:

        logger.error(
            "Router failed: %s",
            e
        )

        raise


# ============================================================
# ETL Node
# ============================================================

def etl_node(state: DataAgentSchema):

    logger.info("ETL agent started")

    try:

        user_message = state.messages[-1].content

        response = etl_analyst.invoke(
            {
                "messages": [
                    HumanMessage(content=user_message)
                ]
            }
        )

        # Get messages returned by ETL agent
        child_messages = response.get(
            "messages",
            []
        )

        logger.debug(
            "ETL agent returned %d messages",
            len(child_messages)
        )

        # Find the final useful response
        final_message = None

        for message in reversed(child_messages):

            if not hasattr(
                message,
                "content"
            ):
                continue

            if not message.content:
                continue

            # Don't return tool-call messages as final answer
            if getattr(
                message,
                "tool_calls",
                None
            ):
                continue

            final_message = message.content

            break

        if final_message is None:

            logger.warning(
                "ETL agent completed without a final response"
            )

            final_message = (
                "The ETL operation was completed successfully."
            )

        # Agent-generated response should be AIMessage
        state.messages = state.messages + [
            AIMessage(
                content=final_message
            )
        ]

        logger.info(
            "ETL agent completed"
        )

        return state

    except Exception as e:

        logger.error(
            "ETL agent failed: %s",
            e
        )

        raise


# ============================================================
# SQL Node
# ============================================================

def sql_node(state: DataAgentSchema):

    logger.info("SQL agent started")

    try:

        user_message = state.messages[-1].content

        input_schema = {

            "messages": [],

            "user_question": user_message,

            "curated_ques": "",

            "prompt_query_context": "",

            "generated_sql_query": "",

            "is_safe": "No",

            "comments": "",

            "sql_query_execution_result": "",

            "final_answer": "",
        }

        logger.debug(
            "Invoking SQL Analyst"
        )

        response = sql_analyst.invoke(
            input_schema
        )

        # ----------------------------------------------------
        # SQL agent returns a state dictionary / model
        # ----------------------------------------------------

        if hasattr(
            response,
            "model_dump"
        ):

            response_data = response.model_dump()

        elif isinstance(
            response,
            dict
        ):

            response_data = response

        else:

            logger.warning(
                "SQL agent returned unexpected response type: %s",
                type(response).__name__
            )

            response_data = {}

        # ----------------------------------------------------
        # Get final answer
        # ----------------------------------------------------

        final_answer = response_data.get(
            "final_answer",
            ""
        )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if not final_answer:

            logger.warning(
                "SQL agent completed without a final answer"
            )

            final_answer = (
                "The SQL operation was completed, "
                "but no final answer was returned."
            )

        # Agent-generated response should be AIMessage
        state.messages = state.messages + [
            AIMessage(
                content=final_answer
            )
        ]

        logger.info(
            "SQL agent completed"
        )

        return state

    except Exception as e:

        logger.error(
            "SQL agent failed: %s",
            e
        )

        raise


# ============================================================
# Routing Edge
# ============================================================

def route_edge(state: DataAgentSchema) -> str:

    route = state.route_response

    logger.info(
        "Evaluating route: %s",
        route
    )

    if route == "sql":

        logger.info(
            "Routing request to SQL node"
        )

        return "sql_node"

    elif route == "etl":

        logger.info(
            "Routing request to ETL node"
        )

        return "etl_node"

    else:

        logger.error(
            "Invalid route response: %s",
            route
        )

        raise ValueError(
            f"Invalid route response: {route}"
        )


# ============================================================
# Build Graph
# ============================================================

logger.info(
    "Building Data Agent graph"
)

data_agent_graph = StateGraph(
    DataAgentSchema
)


data_agent_graph.add_node(
    "router_node",
    router_node
)

data_agent_graph.add_node(
    "etl_node",
    etl_node
)

data_agent_graph.add_node(
    "sql_node",
    sql_node
)


# ============================================================
# Graph Edges
# ============================================================

data_agent_graph.add_edge(
    START,
    "router_node"
)


data_agent_graph.add_conditional_edges(
    "router_node",
    route_edge,
    {
        "sql_node": "sql_node",
        "etl_node": "etl_node",
    }
)


# End after SQL / ETL

data_agent_graph.add_edge(
    "sql_node",
    END
)

data_agent_graph.add_edge(
    "etl_node",
    END
)


# ============================================================
# Compile
# ============================================================

data_agent = data_agent_graph.compile()

logger.info(
    "Data Agent graph compiled successfully"
)


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    logger.info(
        "Starting Data Agent standalone test"
    )

    try:

        response = data_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "What are the different "
                            "payment methods we have "
                            "in our databases?"
                        )
                    )
                ],

                "route_response": "",
            }
        )

        logger.info(
            "Data Agent standalone test completed"
        )

        print("\n")
        print("=" * 70)
        print("DATA AGENT RESPONSE")
        print("=" * 70)

        for message in response["messages"]:

            if hasattr(
                message,
                "content"
            ) and message.content:

                print(message.content)

    except Exception as e:

        logger.error(
            "Data Agent standalone test failed: %s",
            e
        )

        raise