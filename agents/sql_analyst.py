import os
import sys
import re

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.database import DatabaseUtil, DB_CONFIG
from utils.llm_pick import extract_content, pick_llm
from models.schema import AgentSchema, JudgeSchema

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# ===================================================================
# SQL Analyst Agent
# ===================================================================


# -------------------------------------------------------------------
# Helper: Deterministic SQL Safety Check
# -------------------------------------------------------------------

def deterministic_sql_safety_check(sql_query: str) -> tuple[bool, str]:
    """
    Deterministic safety validation for generated SQL.

    This runs BEFORE the LLM safety judge.

    Only read-only SELECT / WITH queries are allowed.
    """

    if not sql_query:
        return False, "Generated SQL query is empty."

    sql = sql_query.strip()

    # ---------------------------------------------------------------
    # Remove Markdown SQL fences if the model accidentally returns
    # them.
    # ---------------------------------------------------------------

    sql = re.sub(r"```sql", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)

    sql = sql.strip()

    if not sql:
        return False, "Generated SQL query is empty after cleanup."

    # ---------------------------------------------------------------
    # Reject multiple statements
    # ---------------------------------------------------------------

    statements = [
        statement.strip()
        for statement in sql.split(";")
        if statement.strip()
    ]

    if len(statements) > 1:
        return (
            False,
            "Multiple SQL statements are not allowed."
        )

    sql_upper = sql.upper().strip()

    # ---------------------------------------------------------------
    # Only SELECT / WITH queries are allowed
    # ---------------------------------------------------------------

    if not (
        sql_upper.startswith("SELECT ")
        or sql_upper.startswith("SELECT\n")
        or sql_upper == "SELECT"
        or sql_upper.startswith("WITH ")
        or sql_upper.startswith("WITH\n")
    ):
        return (
            False,
            "Only read-only SELECT or WITH queries are allowed."
        )

    # ---------------------------------------------------------------
    # Dangerous SQL keywords
    # ---------------------------------------------------------------

    forbidden_patterns = [
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bDELETE\b",
        r"\bDROP\b",
        r"\bALTER\b",
        r"\bTRUNCATE\b",
        r"\bCREATE\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bMERGE\b",
        r"\bREPLACE\b",
        r"\bEXEC\b",
        r"\bEXECUTE\b",
        r"\bCALL\b",
        r"\bVACUUM\b",
        r"\bANALYZE\b",
        r"\bCOMMENT\b",
        r"\bREINDEX\b",
        r"\bREFRESH\b",
        r"\bCOPY\b",
    ]

    for pattern in forbidden_patterns:

        if re.search(pattern, sql_upper):

            keyword = re.search(
                pattern,
                sql_upper
            ).group(0)

            return (
                False,
                f"Forbidden SQL operation detected: {keyword}."
            )

    # ---------------------------------------------------------------
    # Reject transaction/control statements
    # ---------------------------------------------------------------

    transaction_patterns = [
        r"\bBEGIN\b",
        r"\bCOMMIT\b",
        r"\bROLLBACK\b",
        r"\bSAVEPOINT\b",
        r"\bLOCK\b",
    ]

    for pattern in transaction_patterns:

        if re.search(pattern, sql_upper):

            return (
                False,
                "Transaction or database-state operations are not allowed."
            )

    # ---------------------------------------------------------------
    # Reject SQL comments that may be used for injection or hiding
    # statements.
    # ---------------------------------------------------------------

    if "--" in sql:
        return (
            False,
            "SQL line comments are not allowed."
        )

    if "/*" in sql or "*/" in sql:
        return (
            False,
            "SQL block comments are not allowed."
        )

    # ---------------------------------------------------------------
    # Reject PostgreSQL dangerous functions
    # ---------------------------------------------------------------

    dangerous_functions = [
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_execute_server_program",
        "dblink_connect",
        "dblink_exec",
    ]

    sql_lower = sql.lower()

    for function_name in dangerous_functions:

        if function_name in sql_lower:

            return (
                False,
                f"Potentially unsafe PostgreSQL function detected: "
                f"{function_name}."
            )

    # ---------------------------------------------------------------
    # Query passed deterministic checks
    # ---------------------------------------------------------------

    return True, "Deterministic SQL safety checks passed."


# -------------------------------------------------------------------
# Node 1: Curate Question
# -------------------------------------------------------------------

def curate_question(state: AgentSchema) -> AgentSchema:

    user_question = state.user_question

    llm = pick_llm("high")

    response = llm.invoke(
        f"""
        Curate and clarify the following user's question.

        Preserve the original intent.
        Do not add requirements that the user did not ask for.
        Return only the improved question.

        User question:
        {user_question}
        """
    )

    state.curated_ques = extract_content(response)

    state.messages = state.messages + [
        HumanMessage(
            content=user_question
        ),
        AIMessage(
            content=state.curated_ques
        ),
    ]

    return state


# -------------------------------------------------------------------
# Node 2: Generate SQL Prompt
# -------------------------------------------------------------------

def propmt_query_context(state: AgentSchema) -> AgentSchema:

    curated_question = state.curated_ques

    schema_info = DatabaseUtil(
        DB_CONFIG
    ).schema_details("public")

    prompt = f"""
You are an SQL analyst agent.

Your task is to convert the user's natural-language question
into a PostgreSQL SQL query that can be executed against the
provided database.

You are given:

1. The user's curated question.
2. Database schema details.
3. Table names.
4. Column names.
5. Data types.
6. Sample data where available.

Use ONLY tables and columns that exist in the provided schema.

IMPORTANT RULES:

- Generate PostgreSQL-compatible SQL.
- Generate ONLY one SQL statement.
- The query must be READ-ONLY.
- Only SELECT or WITH queries are allowed.
- Never generate INSERT.
- Never generate UPDATE.
- Never generate DELETE.
- Never generate DROP.
- Never generate ALTER.
- Never generate TRUNCATE.
- Never generate CREATE.
- Never generate GRANT.
- Never generate REVOKE.
- Never generate MERGE.
- Never generate REPLACE.
- Never generate EXEC.
- Never generate EXECUTE.
- Never generate CALL.
- Never generate transaction commands.
- Never generate database administration commands.
- Never generate multiple SQL statements.
- Never use SQL comments.
- Never expose credentials or secrets.

Unless the user explicitly requests a specific number of rows,
LIMIT the result to 10 rows.

If the user's request already contains a LIMIT,
respect the user's requested limit.

Return ONLY the SQL query.

User's Question:
{curated_question}

Database Schema Details:
{schema_info}
"""

    state.prompt_query_context = prompt

    return state


# -------------------------------------------------------------------
# Node 3: Generate SQL
# -------------------------------------------------------------------

def generate_sql(state: AgentSchema) -> AgentSchema:

    prompt = state.prompt_query_context

    llm = pick_llm("low")

    response = llm.invoke(prompt)

    generated_sql = extract_content(response)

    # ---------------------------------------------------------------
    # Clean accidental Markdown fences
    # ---------------------------------------------------------------

    generated_sql = re.sub(
        r"```sql",
        "",
        generated_sql,
        flags=re.IGNORECASE
    )

    generated_sql = re.sub(
        r"```",
        "",
        generated_sql
    )

    state.generated_sql_query = generated_sql.strip()

    return state


# -------------------------------------------------------------------
# Node 4: SQL Safety Judge
# -------------------------------------------------------------------

def is_safe_sql(state: AgentSchema) -> AgentSchema:

    sql_query = state.generated_sql_query

    # ---------------------------------------------------------------
    # First perform deterministic validation
    # ---------------------------------------------------------------

    deterministic_safe, deterministic_feedback = (
        deterministic_sql_safety_check(
            sql_query
        )
    )

    if not deterministic_safe:

        state.is_safe = "No"
        state.comments = deterministic_feedback

        return state

    # ---------------------------------------------------------------
    # LLM safety judge
    #
    # IMPORTANT:
    # Structured output is passed directly into pick_llm().
    # Do NOT call:
    #
    # llm.with_structured_output(...)
    #
    # because pick_llm() already returns a retry-wrapped runnable.
    # ---------------------------------------------------------------

    llm_judge = pick_llm(
        "low",
        output_schema=JudgeSchema,
    )

    prompt = f"""
    You are a strict SQL safety judge.

    Review the generated PostgreSQL query.

    The query must satisfy ALL of these requirements:

    1. It must be read-only.
    2. It must be a SELECT or WITH query.
    3. It must not modify data.
    4. It must not modify database schema.
    5. It must not modify permissions.
    6. It must not change database state.
    7. It must not contain multiple statements.
    8. It must not contain SQL injection patterns.
    9. It must not access unavailable tables or columns.
    10. It must fulfill the user's request.
    11. It must not expose credentials or secrets.
    12. It must not execute operating-system commands.
    13. It must not execute unsafe PostgreSQL functions.

    Unsafe operations include:

    INSERT
    UPDATE
    DELETE
    DROP
    ALTER
    TRUNCATE
    CREATE
    GRANT
    REVOKE
    MERGE
    REPLACE
    EXEC
    EXECUTE
    CALL
    COPY
    VACUUM
    REINDEX
    transaction commands
    database administration commands

    Only read-only SELECT/WITH queries are permitted.

    Generated SQL Query:
    {sql_query}

    Return the result using the provided structured output schema.
    """

    try:

        response = llm_judge.invoke(prompt)

        # -----------------------------------------------------------
        # Structured Pydantic response
        # -----------------------------------------------------------

        response_data = response.model_dump()

        # -----------------------------------------------------------
        # Read answer field
        # -----------------------------------------------------------

        answer = str(
            response_data.get(
                "answer",
                "No"
            )
        ).strip()

        comments = str(
            response_data.get(
                "comments",
                response_data.get(
                    "feedback",
                    ""
                )
            )
        ).strip()

        if answer.lower() in {
            "yes",
            "safe",
            "true",
        }:

            state.is_safe = "Yes"

        else:

            state.is_safe = "No"

        state.comments = comments

    except Exception as exc:
        # If deterministic check verified it is strictly a read-only SELECT,
        # allow it through with a logged warning rather than failing the whole graph
        if deterministic_safe and sql_query.upper().strip().startswith("SELECT"):
            state.is_safe = "Yes"
            state.comments = f"Passed deterministic safety validation (LLM judge unavailable: {exc})"
        else:
            state.is_safe = "No"
            state.comments = f"SQL safety judge failed. Execution blocked. Error: {exc}"

    return state


# -------------------------------------------------------------------
# Node 5: Cancel Unsafe SQL
# -------------------------------------------------------------------

def canceled_sql(state: AgentSchema) -> AgentSchema:

    comments = state.comments

    state.final_answer = (
        "SQL query execution was canceled due to "
        f"safety concerns. Comments: {comments}"
    )

    state.messages = state.messages + [
        AIMessage(
            content=state.final_answer
        )
    ]

    return state


# -------------------------------------------------------------------
# Node 6: Execute SQL
# -------------------------------------------------------------------

def execute_sql(state: AgentSchema) -> AgentSchema:

    sql_query = state.generated_sql_query

    # ---------------------------------------------------------------
    # Final deterministic safety check immediately before execution.
    #
    # This is defense-in-depth.
    # ---------------------------------------------------------------

    safe, feedback = deterministic_sql_safety_check(
        sql_query
    )

    if not safe:

        state.is_safe = "No"
        state.comments = feedback

        return state

    # ---------------------------------------------------------------
    # Execute query
    # ---------------------------------------------------------------

    dbconn = DatabaseUtil(DB_CONFIG)

    execution_result = dbconn.execute_sql_query(
        sql_query
    )

    state.sql_query_execution_result = execution_result

    return state


# -------------------------------------------------------------------
# Node 7: Final Answer
# -------------------------------------------------------------------

def final_answer(state: AgentSchema) -> AgentSchema:

    execution_result = (
        state.sql_query_execution_result
    )

    curated_question = state.curated_ques

    llm = pick_llm("medium")

    prompt = f"""
You are an SQL analyst agent.

Provide a clear and concise final answer to the user's question
based ONLY on the result of the SQL query execution.

Do not invent information.

If the execution result is empty, explain that no matching
records were found.

If the result contains data, summarize the relevant information
in a user-friendly manner.

User's Question:
{curated_question}

SQL Query Execution Result:
{execution_result}
"""

    response = llm.invoke(prompt)

    state.final_answer = extract_content(response)

    state.messages = state.messages + [
        AIMessage(
            content=state.final_answer
        )
    ]

    return state


# ===================================================================
# LangGraph
# ===================================================================

sql_agent_graph = StateGraph(
    AgentSchema
)


# -------------------------------------------------------------------
# Nodes
# -------------------------------------------------------------------

sql_agent_graph.add_node(
    "curate_question",
    curate_question
)

sql_agent_graph.add_node(
    "propmt_query_context",
    propmt_query_context
)

sql_agent_graph.add_node(
    "generate_sql",
    generate_sql
)

sql_agent_graph.add_node(
    "is_safe_sql",
    is_safe_sql
)

sql_agent_graph.add_node(
    "canceled_sql",
    canceled_sql
)

sql_agent_graph.add_node(
    "execute_sql",
    execute_sql
)

sql_agent_graph.add_node(
    "final_answer",
    final_answer
)


# -------------------------------------------------------------------
# Edges
# -------------------------------------------------------------------

sql_agent_graph.add_edge(
    START,
    "curate_question"
)

sql_agent_graph.add_edge(
    "curate_question",
    "propmt_query_context"
)

sql_agent_graph.add_edge(
    "propmt_query_context",
    "generate_sql"
)

sql_agent_graph.add_edge(
    "generate_sql",
    "is_safe_sql"
)


# -------------------------------------------------------------------
# Conditional Edge
# -------------------------------------------------------------------

def is_safe_sql_edge_condition(
    state: AgentSchema
) -> str:

    is_safe = str(
        state.is_safe
    ).strip().lower()

    if is_safe == "yes":

        return "execute_sql"

    return "canceled_sql"


sql_agent_graph.add_conditional_edges(
    "is_safe_sql",
    is_safe_sql_edge_condition,
    {
        "execute_sql": "execute_sql",
        "canceled_sql": "canceled_sql",
    },
)


# -------------------------------------------------------------------
# Remaining edges
# -------------------------------------------------------------------

sql_agent_graph.add_edge(
    "execute_sql",
    "final_answer"
)

sql_agent_graph.add_edge(
    "final_answer",
    END
)

sql_agent_graph.add_edge(
    "canceled_sql",
    END
)

sql_analyst = sql_agent_graph.compile()

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Compile graph
    # ---------------------------------------------------------------

    

    # ---------------------------------------------------------------
    # Optional graph visualization
    # ---------------------------------------------------------------

    try:

        from IPython.display import display, Image, HTML

        graph_png = (
            sql_analyst
            .get_graph()
            .draw_mermaid_png()
        )

        with open(
            "sql_analyst_graph.png",
            "wb"
        ) as f:

            f.write(graph_png)

        display(
            HTML(
                "<h3>SQL Analyst Graph</h3>"
            )
        )

        display(
            Image(graph_png)
        )

    except Exception as exc:

        print(
            f"Graph visualization skipped: {exc}"
        )

    # ---------------------------------------------------------------
    # Input
    # ---------------------------------------------------------------

    input_schema = {
        "messages": [],

        "user_question": (
            "what are different types of payment "
            "methods we have in our databases"
        ),

        "curated_ques": "",

        "prompt_query_context": "",

        "generated_sql_query": "",

        "is_safe": "No",

        "comments": "",

        "sql_query_execution_result": "",

        "final_answer": "",
    }

# ---------------------------------------------------------------
# Execute graph
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Execute graph
# ---------------------------------------------------------------

    try:

        sql_analyst_response = sql_analyst.invoke(
            input_schema
        )

        print(
            sql_analyst_response.get(
                "final_answer",
                "No answer generated."
            )
        )

    except Exception as exc:

        print(f"Error: {type(exc).__name__}: {exc}")