import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
# Logging is configured centrally by the application.
# This module only creates a module-level logger.
logger = logging.getLogger(__name__)


class ETLTools:
    """
    Utility class for ETL-related operations.

    Provides functionality to:
    - Extract data from APIs.
    - Save extracted data as CSV, JSON, or Parquet.
    - Resolve input data files.
    - Load data for transformation context.
    - Execute generated transformation code.
    """

    def __init__(self):
        """
        Initialize ETL tools.

        No initialization logic is currently required.
        """
        logger.debug("ETLTools initialized")

    # -----------------------------------------------------------------------
    # API Response Handling
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_records(payload):
        """
        Normalize different API response structures into a list of records.

        Supported response structures include:
        - Direct list of records.
        - Dictionaries containing results/data/items/records/rows.
        - A dictionary representing a single record.
        - Other payload types converted into a single-item list.

        Args:
            payload: JSON-decoded API response.

        Returns:
            list: Normalized list of records.
        """

        if payload is None:
            logger.debug("API response payload is empty")
            return []

        # API directly returned a list of records.
        if isinstance(payload, list):
            logger.debug(
                "API response contains %d records",
                len(payload),
            )
            return payload

        # API returned a dictionary.
        if isinstance(payload, dict):

            # Look for common keys used to store record collections.
            for key in ("results", "data", "items", "records", "rows"):
                value = payload.get(key)

                if isinstance(value, list):
                    logger.debug(
                        "Records found under API response key: %s",
                        key,
                    )
                    return value

            # If the dictionary contains scalar values, treat it as
            # a single record.
            if any(
                not isinstance(v, (dict, list))
                for v in payload.values()
            ):
                logger.debug("API response treated as a single record")
                return [payload]

            # Otherwise, treat the complete dictionary as one record.
            logger.debug(
                "API dictionary treated as a single record"
            )
            return [payload]

        # Handle non-list/non-dictionary payloads as a single record.
        logger.debug(
            "API response of type %s treated as a single record",
            type(payload).__name__,
        )

        return [payload]

    # -----------------------------------------------------------------------
    # Input File Resolution
    # -----------------------------------------------------------------------

    @staticmethod
    def resolve_input_file(file_path: str) -> Path:
        """
        Resolve the input data file.

        If the supplied path is relative, it is resolved against
        the project root. If a directory is supplied, the latest
        supported data file in that directory is selected.

        Supported formats:
        - CSV
        - JSON
        - Parquet

        Args:
            file_path: File or directory path.

        Returns:
            Path: Resolved input file.

        Raises:
            FileNotFoundError: If no supported input file is found.
        """

        logger.info("Resolving input file")

        path = Path(file_path)

        # Resolve relative paths against the project root.
        project_root = Path(__file__).resolve().parent.parent

        if not path.is_absolute():
            path = project_root / path

        logger.debug("Resolved input path: %s", path)

        # If the supplied path is a directory, search inside it.
        # Otherwise, search in its parent directory.
        search_folder = path if path.is_dir() else path.parent

        supported_extensions = {
            ".csv",
            ".json",
            ".parquet",
        }

        # Find supported files and sort by modification time so that
        # the latest available file is selected.
        input_files = sorted(
            (
                candidate
                for candidate in search_folder.iterdir()
                if candidate.is_file()
                and candidate.suffix.lower() in supported_extensions
            ),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )

        if not input_files:
            logger.warning(
                "No supported input file found for path: %s",
                file_path,
            )

            raise FileNotFoundError(
                f"No supported input file found for '{file_path}'"
            )

        selected_file = input_files[0]

        logger.info(
            "Input file resolved successfully: %s",
            selected_file,
        )

        return selected_file

    # -----------------------------------------------------------------------
    # Extract and Load
    # -----------------------------------------------------------------------

    def extract_load(
        self,
        url: str,
        output_folder: str,
        format: str,
    ):
        """
        Extract data from an API endpoint and save it to the output folder.

        Args:
            url: API endpoint from which data is extracted.
            output_folder: Destination folder for extracted data.
            format: Output format: csv, json, or parquet.

        Returns:
            str: Status message describing the operation result.
        """

        logger.info("Starting API extraction")

        project_root = Path(__file__).resolve().parent.parent
        output_folder = str(project_root / output_folder)

        try:
            # Send GET request to the API.
            # The URL itself is intentionally not logged because it may
            # potentially contain sensitive query parameters.
            response = requests.get(
                url,
                timeout=30,
            )

            response.raise_for_status()

            logger.info("API request completed successfully")

            # Parse API response as JSON.
            payload = response.json()

            # Normalize the API response into records.
            records = self._get_records(payload)

            logger.info(
                "Extracted %d record(s) from API response",
                len(records),
            )

            if not records:
                logger.warning("No data found in API response")
                return "No data found in the API response."

            # Convert JSON records into a Pandas DataFrame.
            df = pd.json_normalize(
                records,
                sep="_",
            )

            logger.debug(
                "Created DataFrame with %d row(s) and %d column(s)",
                len(df),
                len(df.columns),
            )

            # Create output directory if it does not already exist.
            os.makedirs(
                output_folder,
                exist_ok=True,
            )

            # Generate a timestamp-based output filename.
            timestamp = datetime.now().strftime(
                "%Y%m%d%H%M%S"
            )

            filename = os.path.join(
                output_folder,
                f"extract_{timestamp}.{format}",
            )

            # Save the extracted data using the requested format.
            if format == "csv":
                df.to_csv(
                    filename,
                    index=False,
                )

            elif format == "json":
                df.to_json(
                    filename,
                    orient="records",
                    lines=True,
                )

            elif format == "parquet":
                df.to_parquet(
                    filename,
                    index=False,
                )

            else:
                logger.warning(
                    "Unsupported output format requested: %s",
                    format,
                )

                return f"unsupported format: {format}"

            logger.info(
                "API extraction completed successfully"
            )

            logger.info(
                "Extracted data saved successfully"
            )

            return (
                f"Data successfully extracted and saved to {filename}"
            )

        except requests.exceptions.RequestException as e:
            logger.error(
                "API extraction failed: %s",
                e,
            )

            return f"Failed to extract data: {e}"

        except Exception as e:
            logger.error(
                "Unexpected error during API extraction: %s",
                e,
            )

            return f"Failed to extract data: {e}"

    # -----------------------------------------------------------------------
    # Transform Context
    # -----------------------------------------------------------------------

    def transform_load_context(
        self,
        file_path: str,
        output_folder: str | None = None,
        output_format: str | None = None,
    ):
        """
        Load an input file and provide a small sample for transformation.

        Currently, this method loads the supported file and returns
        the first three rows. Actual transformation and loading logic
        can be performed by the generated ETL workflow.

        Args:
            file_path: Path to the input data file.
            output_folder: Optional output folder.
            output_format: Optional output format.

        Returns:
            str: First three rows of the input DataFrame or an error message.
        """

        logger.info("Starting ETL transform context")

        try:
            # Resolve the supplied path to the latest supported input file.
            file_path = self.resolve_input_file(file_path)

        except FileNotFoundError as error:
            logger.warning(
                "Unable to resolve ETL input file: %s",
                error,
            )

            return str(error)

        file_extension = file_path.suffix.lower()

        logger.info(
            "Loading input file with format: %s",
            file_extension,
        )

        try:
            # Load the input file according to its extension.
            if file_extension == ".csv":
                df = pd.read_csv(file_path)

            elif file_extension == ".json":
                df = pd.read_json(
                    file_path,
                    lines=True,
                )

            elif file_extension == ".parquet":
                df = pd.read_parquet(file_path)

            else:
                logger.warning(
                    "Unsupported input file format: %s",
                    file_extension,
                )

                return (
                    f"unsupported file format: {file_extension}"
                )

            logger.info(
                "Input file loaded successfully"
            )

            logger.debug(
                "Loaded DataFrame with %d row(s) and %d column(s)",
                len(df),
                len(df.columns),
            )

            # Return a small sample to provide context to the
            # transformation/LLM workflow.
            top_3_rows = str(df.head(3))

            logger.info(
                "ETL transform context generated successfully"
            )

            return top_3_rows

        except Exception as e:
            logger.error(
                "Failed to load input file: %s",
                e,
            )

            return f"Failed to load input file: {e}"

    # -----------------------------------------------------------------------
    # Code Execution
    # -----------------------------------------------------------------------

    def execute_code(self, code: str):
        """
        Execute the provided transformation code.

        NOTE:
            This currently uses Python exec().
            LLM-generated code should not be executed this way in
            a production environment without proper sandboxing or
            validation.

        Args:
            code: Python code to execute.

        Returns:
            str: Execution status message.
        """

        logger.info("Starting ETL code execution")

        try:
            # Execute the supplied code.
            # Security hardening for this execution path is tracked
            # separately and should be addressed before production use.
            exec(code)

            logger.info(
                "ETL code executed successfully"
            )

            return "code executed successfully."

        except Exception as e:
            logger.error(
                "ETL code execution failed: %s",
                e,
            )

            return (
                f"Failed to execute the provided code: {e}"
            )


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logger.info("Starting ETLTools standalone test")

    obj = ETLTools()

    # Example API extraction:
    #
    # print(
    #     obj.extract_load(
    #         "https://pokeapi.co/api/v2/pokemon/",
    #         "data/extract",
    #         "csv",
    #     )
    # )

    path = (
        r"D:\agentic ai\Agentic AI-Based Multi-Source Data Retrieval "
        r"and Stakeholder Intelligence Platform\data\extract"
        r"\extract_20260913183547.csv"
    )

    try:
        result = obj.transform_load_context(path)

        print(result)

        logger.info(
            "ETLTools standalone test completed successfully"
        )

    except Exception as e:
        logger.error(
            "ETLTools standalone test failed: %s",
            e,
        )