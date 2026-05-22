#!/usr/bin/env python

import json
import sys

def transform_entry(entry):
    """Move m, sigma, epsilon_k into a model_record dict"""
    transformed = {}

    for key, value in entry.items():
        if key in ("m", "sigma", "epsilon_k"):
            continue  # will be moved into model_record
        transformed[key] = value

    transformed["model_record"] = {
        "m": entry["m"],
        "sigma": entry["sigma"],
        "epsilon_k": entry["epsilon_k"]
    }

    return transformed


def transform_json(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    # Handle both a list of entries or a single entry
    if isinstance(data, list):
        transformed = [transform_entry(entry) for entry in data]
    elif isinstance(data, dict):
        transformed = transform_entry(data)
    else:
        raise ValueError("Unexpected JSON structure.")

    with open(output_path, "w") as f:
        json.dump(transformed, f, indent=4)

    print(f"Done! Transformed file saved to: {output_path}")


if __name__ == "__main__":

    input_path  = "parameters.json"
    output_path = "parameters_08.json"
    transform_json(input_path, output_path)