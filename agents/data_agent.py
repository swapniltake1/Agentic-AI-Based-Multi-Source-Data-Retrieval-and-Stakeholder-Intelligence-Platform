from email import message
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from urllib import response
from typing import cast, final
from opentelemetry.metrics import Observation
from utils import llm_pick
from utils.etl_tools import ETLTools
from utils.database import DatabaseUtil, DB_CONFIG
from utils.llm_pick import extract_content, get_base_llm, get_tool_llm, pick_llm
from models.schema import AgentSchema, ETLAgentSchema, JudgeSchema, RouterSchema, DataAgentSchema
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst

#---- Parent agent

llm = get_base_llm("high")


agent_router =  llm.with_structured_output(RouterSchema)

# print(agent_router.invoke("I want to extract the data from an API and save it as a csv file."))

# Data Agent Graph

def router_node(state:DataAgentSchema):

     message = state.messages[-1].content

     route_response_dict = agent_router.invoke(message).model_dump()

     route_response = route_response_dict['answer']

     state.route_response = route_response

     return state

def etl_node(state:DataAgentSchema):

     message = state.messages[-1].content

     response = etl_analyst.invoke({"messages": [HumanMessage(content=f"{message}")]})  

     state.messages = state.messages + [response]

     return state

def sql_node(state:DataAgentSchema):

     message = state.messages[-1].content

     input_schema = {
             "messages": [],
             "user_question": (f"{message}"),
             "curated_ques": "",
             "prompt_query_context": "",   
             "generated_sql_query": "",    
             "is_safe": "No",
             "comments": "",
             "sql_query_execution_result": "",
             "final_answer": ""
     }    

     response = sql_analyst.invoke(input_schema)

     state.messages = state.messages + [response]

     return state


# Graph and nodes
data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)

data_agent_graph.add_edge(START, "router_node")

def route_edge(state: DataAgentSchema) -> str:
     if state.route_response == "sql":
        return "sql_node"
     elif state.route_response == "etl":
        return "etl_node"
     else:
        raise ValueError(f"Invalid route response: {state.route_response}")
        
data_agent_graph.add_conditional_edges("router_node", route_edge, {
    "sql_node": "sql_node",
    "etl_node": "etl_node"
})


data_agent = data_agent_graph.compile()

#Optional graph visualization
from IPython.display import display, Image, HTML
graph_png = (data_agent.get_graph().draw_mermaid_png())
with open("data_agent_graph.png", "wb") as f: f.write(graph_png)


if __name__ == "__main__":

    response = data_agent.invoke(
        {"messages": [HumanMessage(content=f"what are different types of payment methods we have in our databases")],
                    "route_response": "" }
    )

    print(response)
   












