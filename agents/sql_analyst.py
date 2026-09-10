import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.llm_pick import extract_content, pick_llm
from models.schema import AgentSchema

#------------------------------ AI AGENT CODE ------------------------------#

def curate_question(state: AgentSchema) -> AgentSchema:
    user_question = state.user_question   #bcz thisis pydantic model, we can access the attributes directly

    llm = pick_llm("low")

    response = llm.invoke(f"curate the following qustion: {user_question}")

    state.curated_ques = extract_content(response)

    return state



def propmt_query_context(state: AgentSchema) -> AgentSchema:

    curated_ques = state.curated_ques

    llm = pick_llm("low")

    response = llm.invoke(f"generate a detailed prompt with sql db context that will help ai agent to generate sql query for the following question: {curated_ques}")

    state.prompt_query_context = extract_content(response)

    return state
