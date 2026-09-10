import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.database import DatabaseUtil, DB_CONFIG
from utils.llm_pick import extract_content, pick_llm
from models.schema import AgentSchema
from langchain_core.schema import HumanMessage

#------------------------------ AI AGENT CODE ------------------------------#

def curate_question(state: AgentSchema) -> AgentSchema:
    user_question = state.user_question   #bcz thisis pydantic model, we can access the attributes directly

    llm = pick_llm("low")

    response = llm.invoke(f"curate the following qustion: {user_question}")

    state.curated_ques = extract_content(response)
    state.messages = state.messages + [HumanMessage(content=f"{response}")]

    return state



def propmt_query_context(state: AgentSchema) -> AgentSchema:

    curated_question = state.curated_ques

    schema_info = DatabaseUtil(DB_CONFIG).schema_details("public")

    prompt = f"""
             You are an SQL analyst agent. Your task is to convert the user's natural language
             query into Postgres SQL query that can be executed on the database. You are provided
             with the user's original query and the schema details of the database, including
             table names, column names, data types, and sample data for each table so that
             you can understand the structure of the database and generate an accurate SQL query.
             Unless user explicitly asks for specific number of rows, always limit the output to 10 rows.
             Note - Just generate the SQL query without any explanation or additional text because
             this query will be executed directly on the database. So, the output should be SQL
             ready to be executed without any modification.

             User's Original Query: {curated_question}
             Database Schema Details: 
             {schema_info}
            """

    state.prompt_query_context = prompt

    llm = pick_llm("medium")

    generated_sql_query = llm.invoke(prompt)

    state.generated_sql_query = extract_content(generated_sql_query)

    return state


