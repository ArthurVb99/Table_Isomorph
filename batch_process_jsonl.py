"""
Batch Processing Script for Table Structure Recognition
Loads JSONL data, filters by token count, processes with LLM, validates, and outputs valid results.
Uses multi-threading for parallel processing.
"""

import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from track_1_llm_prompt.llm_processor import LLMPromptProcessor
from track_1_llm_prompt.validators import validate_llm_output, compare_cell_count

# Get max tokens from environment variable
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "5000"))


def _save_invalid_output(data: dict, model_name: str, error: str,
                        response: str = None, thinking: str = None) -> tuple[Path, Path]:
    """Save a failed sample's raw response and thinking, when available, and error details."""
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    output_dir = Path(__file__).resolve().parent / "invalid_outputs" / f"{safe_model_name}_transposition"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(str(data.get("filename") or "unknown")).stem
    filename = f"imgid_{data.get('imgid')}_{stem}"
    response_path = output_dir / f"{filename}.response.txt"
    error_path = output_dir / f"{filename}.error.txt"

    if response is not None:
        response_path.write_text(response, encoding="utf-8")
    if thinking:
        (output_dir / f"{filename}.thinking.txt").write_text(thinking, encoding="utf-8")
    error_path.write_text(error, encoding="utf-8")
    return response_path, error_path


def process_single_line(item_data: dict, processor: LLMPromptProcessor, split_focus: str,
                       max_tokens: int, imgids_set: set, output_lock: Lock) -> tuple:
    """
    Process a single JSONL line.

    Args:
        item_data (dict): Dictionary containing line_num and JSON data
        processor (LLMPromptProcessor): LLM processor instance
        split_focus (str): Target split to process
        max_tokens (int): Maximum token limit
        imgids_set (set): Set of already processed imgids
        output_lock (Lock): Thread lock for safe printing

    Returns:
        tuple: (success: bool, line_num: int, imgid: str, result: str or None, error: str or None)
    """
    line_num = item_data['line_num']
    data = item_data['data']
    imgid = data.get('imgid')
    llm_response = None

    try:
        json_str = json.dumps(data)

        # Process only specified split
        if data.get("split") != split_focus:
            error_msg = f"Line {line_num}: Skipping, not in split '{split_focus}'"
            with output_lock:
                print(error_msg)
            return False, line_num, imgid, None, error_msg

        # Check for duplicates
        if imgid in imgids_set:
            error_msg = f"Line {line_num}: Skipping, imgid {imgid} already processed"
            with output_lock:
                print(error_msg)
            return True, line_num, imgid, None, error_msg

        # Check token count
        token_count = processor.count_input_tokens(json_str)
        if token_count > max_tokens:
            error_msg = f"Line {line_num}: Token count {token_count} exceeds limit {max_tokens}, skipping"
            with output_lock:
                print(error_msg)
            return False, line_num, imgid, None, error_msg

        with output_lock:
            print(f"Line {line_num}: Processing ({token_count} tokens)...")

        # Process with LLM
        llm_response = processor.process_template(
            "data_transform_transpose.txt",
            input_data=json_str,
            imgid=data["imgid"],
            split=data["split"],
            filename=data["filename"],
        )

        # Validate LLM output
        model, err = validate_llm_output(llm_response)
        if err:
            error_msg = f"Line {line_num}: Validation failed: {err}"
            response_path, error_path = _save_invalid_output(
                data, str(processor.model), error_msg, str(llm_response)
            )
            # error_msg += f" (raw response: {response_path}; error: {error_path})"
            with output_lock:
                print(error_msg)
            return False, line_num, imgid, None, error_msg

        # Check cell count integrity
        ok = compare_cell_count(data, model.model_dump())
        if not ok:
            error_msg = f"Line {line_num}: Cell count mismatch, skipping"
            response_path, error_path = _save_invalid_output(
                data, str(processor.model), error_msg, str(llm_response)
            )
            error_msg += f" (raw response: {response_path}; error: {error_path})"
            with output_lock:
                print(error_msg)
            return False, line_num, imgid, None, error_msg

        # Return valid result
        output_data = model.model_dump()
        result_str = json.dumps(output_data) + '\n'
        success_msg = f"Line {line_num}: ✓ Valid result"
        with output_lock:
            print(success_msg)
        return True, line_num, imgid, result_str, None

    except json.JSONDecodeError as e:
        error_msg = f"Line {line_num}: Invalid JSON: {e}"
        _save_invalid_output(data, str(processor.model), error_msg, llm_response)
        with output_lock:
            print(error_msg)
        return False, line_num, imgid, None, error_msg
    except Exception as e:
        error_msg = f"Line {line_num}: Processing error: {e}"
        _save_invalid_output(data, str(processor.model), error_msg, llm_response,
                             thinking=getattr(e, "thinking", None))
        with output_lock:
            print(error_msg)
        return False, line_num, imgid, None, error_msg


def process_jsonl_data(jsonl_path: str, output_path: str, processor: LLMPromptProcessor,
                       split_focus: str = "train", max_tokens: int = 5000,
                       max_workers: int = 4, selected_imgids: set = None,
                       stats_path: str = None):
    """
    Process JSONL data with parallel processing.

    Args:
        jsonl_path (str): Path to the JSONL file
        output_path (str): Path to the output JSONL file
        processor (LLMPromptProcessor): LLM processor instance
        split_focus (str): Target split to process
        max_tokens (int): Maximum token limit
        max_workers (int): Maximum number of concurrent threads
        stats_path (str): Path to the stats JSON file (defaults next to jsonl_path)
    """
    output_file = Path(output_path)
    print_lock = Lock()

    # Resume after the highest saved imgid and preserve the cumulative sample count.
    imgids_set = set()
    max_imgid = None
    existing_count = 0
    if output_file.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_count += 1
                    imgid = data.get('imgid')
                    if imgid is not None:
                        imgids_set.add(imgid)
                        if max_imgid is None or imgid > max_imgid:
                            max_imgid = imgid
                except json.JSONDecodeError:
                    continue

    # Collect all items to process
    items_to_process = []
    print(f"Reading JSONL file: {jsonl_path}")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                imgid = data.get('imgid')
                if max_imgid is not None and imgid is not None and imgid <= max_imgid:
                    continue
                if selected_imgids is None or imgid in selected_imgids:
                    items_to_process.append({
                        'line_num': line_num,
                        'data': data
                    })
            except json.JSONDecodeError as e:
                with print_lock:
                    print(f"Line {line_num}: Invalid JSON: {e}")

    print(f"Found {len(items_to_process)} valid items to process")
    print(f"Output file: {output_path}")
    print(f"Using {max_workers} concurrent threads")
    print("=" * 60)

    # Process in parallel
    valid_count = existing_count

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(
                process_single_line,
                item,
                processor,
                split_focus,
                max_tokens,
                imgids_set,
                print_lock
            ): item for item in items_to_process
        }

        # Write results as they complete
        with open(output_path, 'a', encoding='utf-8') as out_file:
            for future in as_completed(future_to_item):
                success, line_num, imgid, result_str, error = future.result()

                if result_str:
                    with print_lock:
                        out_file.write(result_str)
                        out_file.flush()
                if success:
                    valid_count += 1

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Total lines processed: {len(items_to_process)}")
    print(f"Valid results saved: {valid_count}")
    print(f"Output file: {output_path}")

    stats_path = Path(stats_path) if stats_path else Path(jsonl_path).parent / f"{output_file.stem}_stats.json"
    processed_count = existing_count + len(items_to_process)
    stats = {
        "processed_count": processed_count,
        "valid_count": valid_count,
        "valid_rate": valid_count / processed_count if processed_count else 0.0,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Stats file: {stats_path}")


def main():
    """Main function - configure paths here"""

    # Get input path from environment variable
    input_path = os.getenv("PATH_INPUT_DATA_JSONL", "")
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("MODEL_LLM", "gpt-3.5-turbo")
    split_focus = os.getenv("SPLIT_FOCUS", "train")

    # Configure number of threads (adjust based on your system)
    max_threads = int(os.getenv("MAX_THREADS", "4"))
    max_tokens = int(os.getenv("MAX_TOKENS", "5000"))

    if not input_path:
        raise ValueError("PATH_INPUT_DATA_JSONL environment variable not set")

    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Generate output path in same directory
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    output_filename = input_filename.replace('.jsonl', f'{split_focus}_processed_{model}.jsonl')
    output_path = os.path.join(input_dir, output_filename)

    # Initialize LLM processor
    try:
        processor = LLMPromptProcessor(provider=provider, model=model)
        print(f"✓ Initialized {provider} LLM provider with model {model}")
    except Exception as e:
        print(f"Error initializing LLM provider: {e}")
        return

    # Process the JSONL data
    process_jsonl_data(
        input_path,
        output_path,
        processor,
        split_focus=split_focus,
        max_tokens=max_tokens,
        max_workers=max_threads
    )


if __name__ == "__main__":
    main()