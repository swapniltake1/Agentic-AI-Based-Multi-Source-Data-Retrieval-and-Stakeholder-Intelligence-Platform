from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


def get_records(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data", "items", "records", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return [payload]


def stringify_list(values):
    if values is None:
        return ""
    if isinstance(values, list):
        items = []
        for item in values:
            if isinstance(item, dict):
                if "name" in item:
                    items.append(str(item["name"]))
                elif "type" in item and isinstance(item["type"], dict):
                    items.append(str(item["type"].get("name", "")))
                elif "ability" in item and isinstance(item["ability"], dict):
                    items.append(str(item["ability"].get("name", "")))
                else:
                    items.append(str(item))
            else:
                items.append(str(item))
        return "; ".join(filter(None, items))
    return str(values)


def clean_pokemon_record(record):
    if not isinstance(record, dict):
        return {}

    clean = {
        "id": record.get("id"),
        "name": record.get("name"),
        "height": record.get("height"),
        "weight": record.get("weight"),
        "base_experience": record.get("base_experience"),
        "order": record.get("order"),
        "is_default": record.get("is_default"),
    }

    types = record.get("types", [])
    if isinstance(types, list):
        clean["type_names"] = stringify_list([
            item.get("type", {}) for item in types if isinstance(item, dict)
        ])

    abilities = record.get("abilities", [])
    if isinstance(abilities, list):
        clean["ability_names"] = stringify_list([
            item.get("ability", {}) for item in abilities if isinstance(item, dict)
        ])

    stats = record.get("stats", [])
    if isinstance(stats, list):
        stat_parts = []
        for item in stats:
            if isinstance(item, dict):
                stat_name = item.get("stat", {}).get("name")
                base_stat = item.get("base_stat")
                if stat_name or base_stat is not None:
                    stat_parts.append(f"{stat_name}:{base_stat}")
        clean["stats_summary"] = "; ".join(stat_parts)

    sprites = record.get("sprites", {})
    if isinstance(sprites, dict):
        clean["front_default"] = sprites.get("front_default", "")
        clean["front_shiny"] = sprites.get("front_shiny", "")

    return clean


def extract_clean_csv(api_url: str, output_folder: str = "data/extract"):
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    records = get_records(payload)
    if not records:
        raise ValueError("No data found in the API response.")

    cleaned_records = [clean_pokemon_record(record) for record in records]
    cleaned_records = [r for r in cleaned_records if r]

    if not cleaned_records:
        raise ValueError("After cleaning, no valid rows were produced.")

    df = pd.DataFrame(cleaned_records)

    project_root = Path(__file__).resolve().parent
    output_path = project_root / output_folder
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = output_path / f"extract_{timestamp}.csv"
    df.to_csv(file_name, index=False)

    return str(file_name)


if __name__ == "__main__":
    url = "https://pokeapi.co/api/v2/pokemon/1/"
    result = extract_clean_csv(url)
    print(f"Clean file created: {result}")
