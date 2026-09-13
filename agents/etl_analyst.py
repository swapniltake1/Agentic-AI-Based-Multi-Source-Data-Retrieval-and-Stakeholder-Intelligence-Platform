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
from utils.llm_pick import extract_content, get_tool_llm, pick_llm
from models.schema import AgentSchema, ETLAgentSchema, JudgeSchema
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

#---------- ETL AGent

@tool
def extract_load_tool(url:str, output_folder:str, format:str) -> str:
    """
    this toool extract the data from the api(url) and load into the desired flder.

    args:
    url(str): the api endpint from which to extract data.
    output_folder (str): the folder where the extarcted data will be saved.
    format (str): the format which to save the extacted data(csv, json, parquet).

    return:
    str: a message indicating the success or failure of the operation.

    """

    etl_tools = ETLTools()
    return etl_tools.extract_load(url, output_folder, format)

@tool
def transform_load_tool(input_file_path:str, output_folder:str, output_format:str, user_question:str) -> str:
    """
    this tool transform the data from the specified file and loads it into the desired location (output_folder)

    args:
    input_file_path(str): th epath to the file containning the date to be transformed.
    output_folder(str) : the folder where the transformed data will be saved.
    output_format(str): he format in which data will be stored.

    return: 
    str: a mesage indicating the sucess or failure of the operation.

    """

    etl_tools = ETLTools()

    resolved_input_file = etl_tools.resolve_input_file(input_file_path)
    top_3_rows = etl_tools.transform_load_context(
        str(resolved_input_file),
        output_folder,
        output_format,
    )

    llm = pick_llm("high")

    prompt = f"""
    You are a Python Data Analyst who uses Pandas to analyze data.
    You need to provide only the Pandas Code that will help to perform the right ETL operations
    as per the user's question. Do not provide any explanation or comments, only
    the code should be provided. The code should be in a format that can be executed
    in a Python environment with Pandas installed.
    Don't write anything else than Pandas Code. \n

    Create the Pandas Dataframe from the data stored in the file : {resolved_input_file} and then
    write the code to transform and save the data at {output_folder}.
    Here's the user's question: {user_question}\n
    Here's the context of the data you will be analyzing: {top_3_rows}\n
    """

    response = llm.invoke(prompt)
    pandas_code = extract_content(response).strip()

    if pandas_code.startswith("```python"):
        pandas_code = pandas_code[len("```python"):].strip()
    elif pandas_code.startswith("```"):
        pandas_code = pandas_code[len("```"):].strip()

    if pandas_code.endswith("```"):
        pandas_code = pandas_code[:-3].strip()

    results = etl_tools.execute_code(pandas_code)

    return f"the data is trasformed and saved in {output_folder} in {output_format} format \n \n Pandas code executed: \n {pandas_code} \n \n execution results: \n {results} "


tools = [extract_load_tool, transform_load_tool]

llm_bind = get_tool_llm("high", tools)



# agent graph 

def llm_node(state:ETLAgentSchema):

    messages = state.messages

    prompt = f"""
        You are a Python Data Analyst who has access to tools that can extract and load,
        transform and load data. You will be provided with a user's question,
        and you would need to perform the right ETL operations as per the user's question.
        If the operation is performed then inform the user and end the conversation.
        Here's the chat history: {messages}\n
    """

    final_answer = llm_bind.invoke(prompt)

    state.messages = messages + [final_answer]

    return state

def tool_node(state:ETLAgentSchema):
    """ this tool is responsible for the invoking the right tool according to user qustions"""

    tools_result = []

    tools_by_name = {tool.name: tool for tool in tools} 

    tool_calls = state.messages[-1].tool_calls

    for tool_call in tool_calls:

        tool = tools_by_name[tool_call['name']]
        Observation = tool.invoke(tool_call['args'])

        tools_result.append(ToolMessage(content=Observation, tool_call_id = tool_call['id']))

    state.messages = state.messages + tools_result

    return state


# Nodes and edges

etl_analyst_graph =  StateGraph(ETLAgentSchema)
etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

etl_analyst_graph.add_edge(START, "llm_node")

def is_tool_call(state: ETLAgentSchema):
    tool_calls = state.messages[-1].tool_calls

    if tool_calls:
        return "tool_node"
    else:
        return "end"

etl_analyst_graph.add_conditional_edges(
    "llm_node", is_tool_call,
     {
     "tool_node": "tool_node",
     "end": END
     }
)

etl_analyst_graph.add_edge("tool_node", "llm_node")



# TEST CODE
if __name__ == "__main__":
   # llm = get_base_llm("high")
   # llm_bind = llm.bind_tools(tools)
   # print(llm_bind.invoke("i want to extract the data from the endpoint 'https://' in folder 'data/extract/new' and in csv format. can you please do that"))


    
    # Compile
    etl_analyst = etl_analyst_graph.compile()

    # Optional graph visualization
    from IPython.display import display, Image, HTML
    graph_png = (etl_analyst.get_graph().draw_mermaid_png())
    with open("etl_analyst_graph.png", "wb") as f: f.write(graph_png)

    # Final
    #response = etl_analyst.invoke(
    #    {"messages": [HumanMessage(content="i want to extract data the data from the api endpoint 'https://pokeapi.co/api/v2/pokemon/' and save it in 'data/extract' folder and in the csv format. ")]}
    #)  

    response = etl_analyst.invoke(
       {"messages": [
           HumanMessage(content="I want to transform the data stored in the 'data/extract' folder and save the transformed data in the 'data/transform' folder.The transformation should filter the data to show bulbasaur pokemon only.")
       ]}
     )  

    print(response) 

