# Agentic AI-Based Multi-Source Data Retrieval and Stakeholder Intelligence Platform

A Python-based, agent-driven data intelligence project for turning natural-language questions into safe SQL queries against a PostgreSQL database. The system blends LLM-based reasoning with rule-based SQL validation, schema introspection, and a structured data pipeline to support stakeholder intelligence workflows.

## Overview

This project is designed to help users ask business questions in plain English and retrieve insights from a structured transportation and payments dataset. The system attempts to:

- interpret a user request using an LLM,
- generate a PostgreSQL query from a database schema,
- validate that the SQL is read-only and safe to execute,
- run the approved query against a PostgreSQL instance,
- return a final answer grounded in the database results.

The current repository includes:

- agent logic for SQL generation and safety review,
- PostgreSQL database setup and CSV ingestion scripts,
- a schema model used for agent state tracking,
- Gemini-based model selection and API configuration.

## Key Features

- Natural-language-to-SQL workflow
- Deterministic SQL safety checks before execution
- LLM judge layer to assess query safety
- PostgreSQL schema introspection and sample data extraction
- CSV-driven database population for a rides, payments, ratings, users, and vehicles dataset
- Structured Pydantic agent state models for orchestration

## Architecture

The project is organized into a few focused modules:

- [main.py](main.py): current entry point; currently a simple starter script.
- [agents/sql_analyst.py](agents/sql_analyst.py): contains the main SQL analyst workflow, question curation, prompt generation, SQL validation, and execution flow.
- [agents/etl_analyst.py](agents/etl_analyst.py): reserved for ETL-oriented logic and downstream data processing tasks.
- [utils/database.py](utils/database.py): PostgreSQL connection utilities and schema inspection helpers.
- [utils/feed_db.py](utils/feed_db.py): creates the PostgreSQL schema and loads CSV files into the database.
- [utils/llm_pick.py](utils/llm_pick.py): selects the Gemini LLM model tier and resolves API credentials.
- [models/schema.py](models/schema.py): Pydantic schemas for agent state and validation.
- [data/](data/): source CSV files used to populate the database.

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
- LangChain + LangGraph
- Google Gemini via LangChain Google GenAI
- Pydantic
- python-dotenv

## Project Structure

```text
.
├── agents/
│   ├── etl_analyst.py
│   ├── scratch.py
│   └── sql_analyst.py
├── data/
│   ├── payments.csv
│   ├── ratings.csv
│   ├── rides.csv
│   ├── users.csv
│   └── vehicles.csv
├── models/
│   └── schema.py
├── utils/
│   ├── database.py
│   ├── etl_tools.py
│   ├── feed_db.py
│   └── llm_pick.py
├── .env
├── main.py
├── pyproject.toml
├── README.md
├── schema_info.txt
└── ...
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

If the environment is not configured as an editable install, you may also install the dependencies listed in [pyproject.toml](pyproject.toml):

```bash
python -m pip install google-api-core ipython langchain-google-genai psycopg2-binary pydantic python-dotenv
```

## Database Setup

Create the PostgreSQL schema and load the CSV data:

```bash
python utils/feed_db.py
```

This script will:

- create the required tables,
- insert data from the CSV files in [data/](data/),
- validate row counts,
- exit after committing the transactions.

## Running the Project

The current repository is still in an early stage, so the main runtime entry point is minimal. To launch the current starter entry point:

```bash
python main.py
```

The real workflow logic for structured agent-based SQL analysis is primarily contained in the scripts under [agents/](agents/), especially [agents/sql_analyst.py](agents/sql_analyst.py).

## How the Agent Workflow Works

1. A user asks a business question in natural language.
2. The SQL analyst agent curates and clarifies that question.
3. A prompt is built using schema metadata from PostgreSQL.
4. An LLM generates a candidate SQL query.
5. A deterministic safety layer blocks dangerous operations such as `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, and other non-read-only actions.
6. The query is reviewed by a judge agent before execution.
7. Approved queries run against PostgreSQL and the final answer is produced.

## Security Notes

This project intentionally enforces read-only SQL safety checks. The SQL validator rejects:

- data modification queries,
- schema-changing statements,
- transaction control commands,
- multi-statement SQL,
- dangerous PostgreSQL functions,
- SQL comments used to hide malicious logic.

This is a useful pattern for building safe LLM-powered database analytics systems.

## Current Status

This is a functional starter project with core database and agent infrastructure, but it still requires integration work depending on your end-user workflow. In particular:

- the app entry point is still minimal,
- the agent logic is modular but not yet fully exposed through a user-facing interface,
- the database connection depends on an existing PostgreSQL instance and `.env` configuration.

## License

This project does not currently include a license file. If you plan to share or distribute it publicly, add an appropriate open-source license before release.

## Contact / Next Steps

If you want to extend this project further, the most valuable next steps are:

- add a command-line application or web interface,
- expose the SQL analyst as an end-to-end API,
- create a proper dashboard or stakeholder reporting layer,
- expand the ETL pipeline and analytics agents,
- add automated tests for SQL safety and data validation.

