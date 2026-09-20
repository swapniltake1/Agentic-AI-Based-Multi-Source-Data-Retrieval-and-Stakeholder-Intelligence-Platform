# Agentic AI-Based Multi-Source Data Retrieval and Stakeholder Intelligence Platform

A Python-based, agent-driven data intelligence platform for turning natural-language questions into actionable results across SQL analytics and ETL workflows. The system combines LLM-based reasoning, LangGraph orchestration, deterministic SQL safety validation, PostgreSQL schema introspection, API/file-based ETL tools, centralized logging, and a Streamlit chat interface.

## Overview

This project is designed to help users interact with transportation and payments data using natural language instead of writing SQL or manually coordinating ETL steps.

The platform currently provides two specialist workflows:

- **SQL Analyst** — interprets a business question, uses PostgreSQL schema metadata and sample data to generate SQL, validates that the query is safe and read-only, executes the approved query, and produces a final answer.
- **ETL Analyst** — selects extraction and transformation tools for API/file workflows and returns the result to the agent.
- **Data Agent / Parent Router** — classifies an incoming request as `sql` or `etl` and routes it to the appropriate specialist workflow.
- **Streamlit UI** — provides a ChatGPT-style interface for interacting with the Data Agent, including chat history, suggested questions, analyst/model metadata, model health checks, and user-friendly error handling.
- **Centralized logging** — application activity is written to both the console and a rotating `logs/app.log` file.

The current repository includes:

- parent-agent routing with LangGraph,
- specialist SQL and ETL agents,
- deterministic SQL safety validation and an additional LLM safety-judge layer,
- PostgreSQL connection and schema inspection utilities,
- CSV-based database loading,
- API extraction and Pandas-based transformation tools,
- centralized Gemini model configuration with fallback and retry support,
- structured Pydantic state models,
- a Streamlit frontend for end-user interaction.

## Key Features

- Natural-language-to-SQL workflow
- Parent agent that routes requests to SQL or ETL specialists
- Deterministic read-only SQL safety validation before execution
- Structured LLM judge for an additional SQL safety review
- PostgreSQL schema introspection and sample-data context
- CSV-driven PostgreSQL database population
- API extraction and Pandas transformation tools for ETL workflows
- Gemini model tiers with centralized configuration
- Model fallback handling for unavailable/rate-limited models
- Retry handling for temporary service/server failures
- Streamlit chat interface with conversation history
- Suggested business questions from the welcome screen
- Live model status checks for configured Gemini tiers
- Response metadata showing analyst, model tier, and execution time
- Safe frontend error messages while detailed errors are retained in logs
- Centralized rotating file logging for application diagnostics
- Structured Pydantic agent state models

## Architecture

The project is organized into a few focused modules:

<img width="1661" height="717" alt="agentic_ai" src="https://github.com/user-attachments/assets/2b1b6551-ca50-4b05-9c74-5defbc122f5b" />
<img width="1661" height="717" alt="Board (1)" src="https://github.com/user-attachments/assets/53fa45c7-48ba-426e-b0f6-ba849145a5be" />


- [main.py](main.py): lightweight application entry point that initializes logging and confirms the data intelligence application has started.
- [app.py](app.py): Streamlit frontend providing the chat experience, model status panel, suggested questions, response metadata, and request/error handling.
- [agents/data_agent.py](agents/data_agent.py): parent LangGraph workflow that classifies each request and routes it to the SQL or ETL specialist.
- [agents/sql_analyst.py](agents/sql_analyst.py): SQL analyst LangGraph workflow for question curation, schema-aware SQL generation, safety validation, execution, and final answer generation.
- [agents/etl_analyst.py](agents/etl_analyst.py): ETL analyst LangGraph workflow that selects extraction and transformation tools based on the user's request.
- [utils/database.py](utils/database.py): PostgreSQL connection utilities and schema inspection helpers.
- [utils/etl_tools.py](utils/etl_tools.py): ETL extraction/transformation utilities, including API extraction and Pandas-based file processing.
- [utils/feed_db.py](utils/feed_db.py): creates the PostgreSQL schema and loads CSV files into the database.
- [utils/llm_pick.py](utils/llm_pick.py): Gemini model configuration, model-tier selection, API-key resolution, fallback handling, retry handling, and response-content extraction.
- [models/schema.py](models/schema.py): Pydantic schemas used for routing and agent state.
- [logging_config.py](logging_config.py): centralized console and rotating-file logging configuration.
- [data/load/](data/load/): CSV files used to populate the PostgreSQL database.

### Current SQL Analyst Graph

The SQL analyst follows a guarded read-only workflow. Unsafe queries are canceled; approved queries are executed and summarized from their database results.

![SQL analyst graph](sql_analyst_graph.png)

The generated graph is also available as [sql_analyst_graph.png](sql_analyst_graph.png). The ETL graph is available as [etl_analyst_graph.png](etl_analyst_graph.png), and the parent Data Agent graph is available as [data_agent_graph.png](data_agent_graph.png).

### Current Parent Data Agent

The parent agent is now implemented and acts as the orchestration layer for the specialist workflows:

```text
User question
    |
    v
Data Agent / Router
    |
    +--> SQL Analyst --> safe SQL execution --> final answer
    |
    +--> ETL Analyst --> extract / transform --> final result
```

The router uses structured LLM output to classify the request as `sql` or `etl`. The selected specialist is then invoked and its final response is returned to the parent state.

## Streamlit User Interface

The repository now includes a Streamlit-based chat interface in [app.py](app.py).

The UI provides:

- ChatGPT-style conversation flow using Streamlit chat components
- New Chat control with session-based chat history
- Suggested questions for common database-analysis scenarios
- SQL Analyst and ETL Analyst capability cards
- Gemini model status for LOW, MEDIUM, and HIGH tiers
- A **Test All Models** control that sends a real test request to each configured model
- Response metadata such as analyst, model tier, and execution duration
- Friendly user-facing error messages while technical details are written to logs
- A visible Streamlit toolbar with the Deploy action hidden
- Responsive wide layout and custom styling for the application shell

Run the UI with:

```bash
streamlit run app.py
```

> **Note:** the Data Agent currently uses the configured MEDIUM model tier for routing. The model-status panel can test all configured tiers independently.

## Gemini Model Handling

Gemini configuration is centralized in [utils/llm_pick.py](utils/llm_pick.py).

The application defines three logical tiers:

- `LOW`
- `MEDIUM`
- `HIGH`

Each tier maps to a configured Gemini model and reasoning-effort setting. The LLM factory supports:

- centralized model configuration,
- `GEMINI_API_KEY` with `GOOGLE_API_KEY` fallback,
- lower-tier fallback models,
- structured-output support,
- tool-enabled LLMs,
- retries for temporary service/server errors,
- fallback handling for rate limits, quota exhaustion, unavailable models, and related API failures.

Keeping model selection in one module makes it easier to change model configuration without modifying the individual agents.

## Logging and Observability

The application now uses centralized logging through [logging_config.py](logging_config.py).

Logging is configured with:

- console output,
- rotating file output at `logs/app.log`,
- a 5 MB file-size threshold,
- up to 3 rotated backup files,
- a consistent timestamp / level / logger / message format,
- reduced log verbosity for noisy third-party libraries.

The application and agent modules log key lifecycle events such as:

- application startup,
- router initialization and route selection,
- agent execution start/completion,
- model initialization and fallback configuration,
- model health tests,
- SQL/ETL execution failures,
- response extraction,
- frontend request failures.

Sensitive API credentials are not written to the application logs.

## Data Model

The PostgreSQL database is structured around a ride-sharing and stakeholder intelligence domain, including tables such as:

- users
- vehicles
- rides
- payments
- ratings

These tables support queries around rider behavior, driver activity, trip performance, payment histories, and rating trends.

## Tech Stack

- Python 3.12+
- PostgreSQL
- psycopg2
- LangChain
- LangGraph
- Google Gemini via LangChain Google GenAI
- Streamlit
- Pandas
- Requests
- Pydantic
- python-dotenv
- Rotating file logging with Python `logging`

## Project Structure

```text
.
├── agents/
│   ├── __init__.py
│   ├── data_agent.py
│   ├── etl_analyst.py
│   ├── scratch.py
│   └── sql_analyst.py
├── data/
│   └── load/
│       ├── payments.csv
│       ├── ratings.csv
│       ├── rides.csv
│       ├── users.csv
│       └── vehicles.csv
├── models/
│   ├── __init__.py
│   └── schema.py
├── utils/
│   ├── data/
│   │   └── extract/
│   │       └── <generated extraction files>
│   ├── database.py
│   ├── etl_tools.py
│   ├── feed_db.py
│   └── llm_pick.py
├── logs/
│   └── app.log
├── app.py
├── logging_config.py
├── main.py
├── pyproject.toml
├── README.md
├── schema_info.txt
├── data_agent_graph.png
├── etl_analyst_graph.png
├── sql_analyst_graph.png
└── uv.lock
```

## Prerequisites

Before running the project, make sure you have:

1. Python 3.12 or newer
2. PostgreSQL running locally or in a reachable environment
3. A Gemini API key from Google AI Studio or a compatible Google AI project
4. Access to the database credentials for your PostgreSQL instance

## Environment Configuration

Create a `.env` file in the project root with values similar to the following:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_postgres_user
DB_PASSWORD=your_postgres_password
GEMINI_API_KEY=your_gemini_api_key

# Optional fallback
# GOOGLE_API_KEY=your_google_api_key
```

The project uses `python-dotenv` to load environment variables automatically.

## Setup

From the project root, install dependencies:

```bash
python -m pip install -e .
```

The primary dependency definition is maintained in [pyproject.toml](pyproject.toml). The repository also contains [uv.lock](uv.lock) for reproducible dependency resolution when using `uv`.

## Database Setup

Create the PostgreSQL schema and load the CSV data:

```bash
python utils/feed_db.py
```

This script will:

- create the required tables,
- insert data from the CSV files in [data/load/](data/load/),
- validate row counts,
- commit the database changes.

## Running the Project

### Start the Streamlit application

For the user-facing application:

```bash
streamlit run app.py
```

### Run the parent Data Agent directly

To execute the parent router and specialist agents without Streamlit:

```bash
python agents/data_agent.py
```

### Run the SQL Analyst directly

Configure PostgreSQL and the Gemini API key first, then run:

```bash
python agents/sql_analyst.py
```

### Run the ETL Analyst directly

Run:

```bash
python agents/etl_analyst.py
```

### Run the lightweight application entry point

```bash
python main.py
```

The specialist and parent-agent modules include executable examples under their `__main__` blocks.

## How the Agent Workflow Works

### End-to-end routing flow

1. A user submits a business or data-engineering request through the Streamlit UI.
2. [agents/data_agent.py](agents/data_agent.py) receives the request.
3. The parent router uses structured LLM output to classify the request as `sql` or `etl`.
4. The corresponding specialist workflow is invoked.
5. The specialist returns a final AI-generated response to the parent state.
6. The Streamlit layer extracts the final response and displays it with lightweight execution metadata.

### SQL Analyst flow

1. The user asks a business question in natural language.
2. The SQL analyst agent curates and clarifies the question.
3. A prompt is built using PostgreSQL schema metadata and sample data.
4. An LLM generates a candidate SQL query.
5. A deterministic safety layer blocks dangerous operations such as `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, and other non-read-only actions.
6. A structured LLM judge reviews queries that pass the deterministic checks.
7. The approved query is checked again immediately before execution.
8. PostgreSQL executes the query.
9. The results are converted into a final stakeholder-facing answer.

### ETL Analyst flow

1. The LLM receives the user's ETL request and conversation state.
2. It selects an available extraction or transformation tool when required.
3. The selected tool executes the requested operation.
4. The tool result is returned to the LLM.
5. The ETL workflow returns the final result to the parent Data Agent.

## Security Notes

This project intentionally enforces read-only SQL safety checks. The SQL validator rejects:

- data modification queries,
- schema-changing statements,
- transaction control commands,
- multi-statement SQL,
- dangerous PostgreSQL functions,
- SQL comments used to hide malicious logic.

The ETL code-execution utility currently uses Python `exec()` for transformation execution. This path is **not production-safe for untrusted or LLM-generated code** and should be isolated or sandboxed before production deployment.

Frontend exceptions are logged with technical details, while the UI returns a generic failure message to reduce exposure of internal implementation details.

## Current Status

The platform now includes the core multi-agent orchestration and a functional user-facing Streamlit application.

### Implemented

- SQL Analyst LangGraph workflow
- ETL Analyst LangGraph workflow
- Parent Data Agent routing between SQL and ETL
- PostgreSQL schema inspection and CSV loading
- Deterministic SQL safety validation
- LLM-based SQL safety review
- Gemini model-tier configuration
- Model fallback and retry handling
- Streamlit chat interface
- Suggested questions and new-chat session handling
- Model health testing from the UI
- Response metadata for analyst/model/execution time
- Centralized application logging with rotating file output
- User-facing error handling that keeps detailed diagnostics in logs

### Remaining work

- Expand the parent router with more specialist agents as the platform grows
- Strengthen ETL code execution with a secure sandbox
- Add automated unit/integration tests for routing, SQL safety, ETL tools, and database validation
- Add production-grade API deployment and authentication
- Add a stakeholder dashboard/reporting layer

## License

This project does not currently include a license file. If you plan to share or distribute it publicly, add an appropriate open-source license before release.

## Contact / Next Steps

The current architecture is designed to grow from a two-specialist agent platform into a broader stakeholder intelligence system. Natural extensions include additional analytics agents, production APIs, observability, testing, authentication, and dashboard/reporting capabilities.
