"""Pydantic validators for LLM table output

Provides models that mirror the expected HTML-table-like structure used
in the prompt templates and helper functions to validate LLM JSON output.

Usage:
    from track_1_llm_prompt.validators import validate_llm_output, compare_cell_count

    model, err = validate_llm_output(llm_response_json)
    if err:
        print('Validation failed:', err)
    else:
        # model is a parsed Pydantic model
        print('Validated imgid:', model.imgid)

"""
from typing import List, Dict, Tuple, Optional, Union
import json
import re

from pydantic import BaseModel, validator, ValidationError


class CellModel(BaseModel):
    tokens: List[str]
    bbox: List[int]

    @validator("bbox")
    def bbox_must_be_four_ints(cls, v):
        if not isinstance(v, list):
            raise ValueError("bbox must be a list of four integers")
        if len(v) != 4:
            raise ValueError("bbox must contain exactly four numbers: [x1, y1, x2, y2]")
        # coerce numeric strings to ints is intentionally not performed here;
        # input should match the expected numeric types coming from the LLM.
        if not all(isinstance(i, int) for i in v):
            raise ValueError("All bbox values must be integers")
        return v


class HtmlStructureModel(BaseModel):
    tokens: List[str]


class HtmlModel(BaseModel):
    cells: List[CellModel]
    structure: HtmlStructureModel


class LLMTableModel(BaseModel):
    imgid: int
    html: HtmlModel
    split: str
    filename: str


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON object from text that may contain surrounding content.

    Looks for the first complete JSON object (starting with '{' and ending with '}').
    Handles nested braces correctly.

    Args:
        text: The text that may contain JSON

    Returns:
        The extracted JSON string, or None if no valid JSON found
    """
    if not text or not isinstance(text, str):
        return None

    # Find the start of a JSON object
    start_idx = text.find('{')
    if start_idx == -1:
        return None

    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the end of the JSON object
                    return text[start_idx:i+1]

    return None


def validate_llm_output(output: Union[str, Dict]) -> Tuple[Optional[LLMTableModel], Optional[str]]:
    """Validate LLM output (JSON string or dict) against the expected schema.

    Can extract JSON from text that contains surrounding content (e.g., explanations).

    Returns a tuple `(model, error_json)`. If validation succeeds, `model` is
    an instance of `LLMTableModel` and `error_json` is None. If validation
    fails, `model` is None and `error_json` contains the Pydantic errors (JSON).
    """
    try:
        data = json.loads(output) if isinstance(output, str) else output
    except Exception as e:
        # If direct JSON parsing fails, try to extract JSON from text
        if isinstance(output, str):
            json_str = extract_json_from_text(output)
            if json_str:
                try:
                    data = json.loads(json_str)
                except Exception as parse_err:
                    return None, f"Invalid JSON: {parse_err}. Extracted: {json_str[:200]}..."
            else:
                return None, f"Invalid JSON: {e}. No JSON object found in text."
        else:
            return None, f"Invalid JSON: {e}"

    try:
        model = LLMTableModel.parse_obj(data)
    except ValidationError as e:
        return None, e.json()

    return model, None


def compare_cell_count(original: Dict, transformed: Dict) -> bool:
    """Simple integrity check: total number of `cells` before and after.

    Returns True if counts match, False otherwise.
    """
    orig = original.get("html", {}).get("cells", []) if isinstance(original, dict) else []
    trans = transformed.get("html", {}).get("cells", []) if isinstance(transformed, dict) else []
    return len(orig) == len(trans)


def test_json_extraction():
    """Test function to demonstrate JSON extraction from mixed text."""
    test_cases = [
        # Pure JSON
        '{"imgid": 1, "html": {"cells": [], "structure": {"tokens": []}}, "split": "train", "filename": "test.png"}',

        # JSON with surrounding text
        'Here is the transposed table:\n\n{"imgid": 1, "html": {"cells": [], "structure": {"tokens": []}}, "split": "train", "filename": "test.png"}\n\nAs you can see, the transposition worked correctly.',

        # JSON with markdown
        '```json\n{"imgid": 1, "html": {"cells": [], "structure": {"tokens": []}}, "split": "train", "filename": "test.png"}\n```',

        # Multiple JSON objects (should extract first)
        'First: {"imgid": 1} Second: {"imgid": 2}',
    ]

    for i, test_input in enumerate(test_cases, 1):
        extracted = extract_json_from_text(test_input)
        model, err = validate_llm_output(test_input)

        print(f"Test {i}:")
        print(f"  Input: {test_input[:50]}...")
        print(f"  Extracted JSON: {extracted is not None}")
        print(f"  Validation: {'PASS' if model else 'FAIL'}")
        if err:
            print(f"  Error: {err[:100]}...")
        print()


if __name__ == "__main__":
    test_json_extraction()
