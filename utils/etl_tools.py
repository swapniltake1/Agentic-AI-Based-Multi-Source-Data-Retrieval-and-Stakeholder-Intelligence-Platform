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





if __name__ == "__main__":
    obj = ETLTools()
    print(obj.extract_load("https://pokeapi.co/api/v2/pokemon/1/", "data/extract", "csv"))







    