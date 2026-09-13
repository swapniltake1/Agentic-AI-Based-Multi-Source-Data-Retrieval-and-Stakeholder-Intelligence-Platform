import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


class ETLTools:

    def __init__(self):
        pass

    @staticmethod
    def _get_records(payload):
        if payload is None:
            return []

        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            for key in ("results", "data", "items", "records", "rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value

            # If the dict itself looks like a single record, keep it as one row.
            if any(not isinstance(v, (dict, list)) for v in payload.values()):
                return [payload]

            # If multiple dictionary keys exist, treat the dict as a single record.
            return [payload]

        return [payload]

    @staticmethod
    def resolve_input_file(file_path: str) -> Path:
        path = Path(file_path)
        project_root = Path(__file__).resolve().parent.parent

        if not path.is_absolute():
            path = project_root / path

        search_folder = path if path.is_dir() else path.parent

        supported_extensions = {".csv", ".json", ".parquet"}
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
            raise FileNotFoundError(
                f"No supported input file found for '{file_path}'"
            )

        return input_files[0]

    def extract_load(self, url: str, output_folder: str, format: str):
        """
        Extract data from an API endpoint and save it into the desired output folder.

        args:
            url (str): The API endpoint from which to extract the data.
            output_folder (str): The folder where the extracted data will be saved.
            format (str): Output format: csv, json, or parquet.

        Returns:
            str: a message indicating the success or failure of the operation.
        """

        project_root = Path(__file__).resolve().parent.parent
        output_folder = str(project_root / output_folder)

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            records = self._get_records(payload)

            if not records:
                return "No data found in the API response."

            df = pd.json_normalize(records, sep="_")

            os.makedirs(output_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = os.path.join(output_folder, f"extract_{timestamp}.{format}")

            if format == "csv":
                df.to_csv(filename, index=False)
            elif format == "json":
                df.to_json(filename, orient="records", lines=True)
            elif format == "parquet":
                df.to_parquet(filename, index=False)
            else:
                return f"unsupported format: {format}"

            return f"Data successfully extracted and saved to {filename}"
        except requests.exceptions.RequestException as e:
            return f"Failed to extract data: {e}"

    def transform_load_context(
        self,
        file_path: str,
        output_folder: str | None = None,
        output_format: str | None = None,
    ):
        """
        this tool trasform the data from the specified file and loads it into the desired location ( output_folder).

        args: 
        filepath (str): the path to the file containing the data to be transformed.
        output_folder (str): the folder where the transformed data will be saved.

        returns:
        str: a message indicating the success or failure of the operation.
        
        """

        try:
            file_path = self.resolve_input_file(file_path)
        except FileNotFoundError as error:
            return str(error)

        file_extension = file_path.suffix.lower()

        if file_extension == ".csv":
           df = pd.read_csv(file_path)
        elif file_extension == ".json":
           df = pd.read_json(file_path, lines=True)
        elif file_extension ==  ".parquet":
           df = pd.read_parquet(file_path)
        else:
          return f"unsupported file format: {file_extension}"


        top_3_rows = str(df.head(3))

        return top_3_rows


    def execute_code(self, code:str):
        """
        this tool run the code is provided and return output.
        """

        try:
            exec(code)
            return "code executed successfully."

        except Exception as e:
            return f"Failed to execute the provided code: {e}"

    

if __name__ == "__main__":
    obj = ETLTools()
    #print(obj.extract_load("https://pokeapi.co/api/v2/pokemon/", "data/extract", "csv"))

    path = r"D:\agentic ai\Agentic AI-Based Multi-Source Data Retrieval and Stakeholder Intelligence Platform\data\extract\extract_20260913183547.csv"

    print(obj.transform_load_context(path))







    