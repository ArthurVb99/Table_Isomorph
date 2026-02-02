"""
Batch Image Processing Script for Table Transposition
Reads JSONL file, processes images using ImagePromptEditor, and saves transposed images.
Uses multi-threading for parallel processing.
"""

import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from track_2_image_editing.image_editor import ImagePromptEditor


def process_single_image(item_data: dict, output_path: Path, editor: ImagePromptEditor, lock: Lock) -> tuple:
    """
    Process a single image item.

    Args:
        item_data (dict): JSON data for the image
        output_path (Path): Output directory path
        editor (ImagePromptEditor): Image editor instance
        lock (Lock): Thread lock for safe printing

    Returns:
        tuple: (success: bool, line_num: int, imgid: str, result_path: str or None, error: str or None)
    """
    line_num = item_data['line_num']
    data = item_data['data']
    imgid = data.get('imgid')
    split = data.get('split')
    filename = data.get('filename')
    in_dir = item_data.get('in_dir', '')

    try:
        if not all([split, filename]):
            error_msg = f"Line {line_num}: Missing split or filename, skipping"
            with lock:
                print(error_msg)
            return False, line_num, imgid, None, error_msg

        # Construct image path
        image_path = Path(in_dir) / split/ filename

        # Check if image exists
        if not image_path.exists():
            error_msg = f"Line {line_num} (imgid: {imgid}): Image not found: {image_path}"
            with lock:
                print(error_msg)
            return False, line_num, imgid, None, error_msg

        with lock:
            print(f"Line {line_num} (imgid: {imgid}): Processing {image_path}")

        # Process image with table transposition
        output_filename = f"{Path(filename).stem}_transposed{Path(filename).suffix}"
        output_file_path = output_path / output_filename

        # if the output file already exists, skip processing
        if output_file_path.exists():
            with lock:
                print(f"Line {line_num}: ✓ Transposed image already exists: {output_file_path}")
            return True, line_num, imgid, str(output_file_path), None

        result_path = editor.transpose_table_image(
            str(image_path),
            output_filename=str(output_file_path)
        )

        if result_path and Path(result_path).exists():
            success_msg = f"Line {line_num}: ✓ Saved transposed image: {result_path}"
            with lock:
                print(success_msg)
            return True, line_num, imgid, result_path, None
        else:
            error_msg = f"Line {line_num}: ✗ Failed to process image"
            with lock:
                print(error_msg)
            return False, line_num, imgid, None, error_msg

    except Exception as e:
        error_msg = f"Line {line_num}: Processing error: {e}"
        with lock:
            print(error_msg)
        return False, line_num, imgid, None, error_msg


def process_jsonl_images(jsonl_path: str, output_dir: str = "processed_images",
						editor: ImagePromptEditor = None, max_workers: int = 80):
	"""
	Process images from JSONL file by transposing tables using parallel processing.

	Args:
		jsonl_path (str): Path to the JSONL file containing image metadata
		output_dir (str): Directory to save processed images
		editor (ImagePromptEditor): Image editor instance (created if None)
		max_workers (int): Maximum number of concurrent threads
	"""
	# Create output directory if it doesn't exist
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True)

	jsonl_dir = Path(jsonl_path).parent

	# Initialize editor if not provided
	if editor is None:
		editor = ImagePromptEditor()

	# Thread lock for safe printing
	print_lock = Lock()

	# Collect all items to process
	items_to_process = []
	total_lines = 0

	print(f"Reading JSONL file: {jsonl_path}")
	with open(jsonl_path, 'r', encoding='utf-8') as f:
		for line_num, line in enumerate(f, 1):
			line = line.strip()
			if not line:
				continue

			try:
				data = json.loads(line)
				items_to_process.append({
					'line_num': line_num,
					'data': data,
					'in_dir': str(jsonl_dir)
				})
				total_lines = line_num
			except json.JSONDecodeError as e:
				with print_lock:
					print(f"Line {line_num}: Invalid JSON: {e}")

	print(f"Found {len(items_to_process)} valid items to process")
	print(f"Output directory: {output_dir}")
	print(f"Using {max_workers} concurrent threads")
	print("=" * 60)

	# Process images in parallel
	processed_count = 0
	error_count = 0

	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		# Submit all tasks
		future_to_item = {
			executor.submit(process_single_image, item, output_path, editor, print_lock): item
			for item in items_to_process
		}

		# Collect results as they complete
		for future in as_completed(future_to_item):
			success, line_num, imgid, result_path, error = future.result()
			if success:
				processed_count += 1
			else:
				error_count += 1

	print("\n" + "=" * 60)
	print("Processing complete!")
	print(f"Successfully processed: {processed_count} images")
	print(f"Errors: {error_count}")
	print(f"Output directory: {output_path}")
	print(f"Total items processed: {processed_count + error_count}")


def main():
    """Main function - configure paths here"""

    # Get input path from environment variable
    jsonl_file = os.getenv("PATH_INPUT_DATA_JSONL_VLM", "")
    provider = os.getenv("VLM_PROVIDER", "openai")
    model = os.getenv("MODEL", "gpt-5")
	# Configure number of threads (adjust based on your system)
    max_threads = int(os.getenv("MAX_THREADS", "40"))
    
    if not jsonl_file:
        print("Error: PATH_INPUT_DATA_JSONL environment variable not set")
        return

    # Configure output directory
    jsonl_dir = Path(jsonl_file).parent
    output_directory = jsonl_dir / f"processed_images_{provider}_{model}"

    # Check if JSONL file exists
    if not os.path.exists(jsonl_file):
        print(f"Error: JSONL file not found: {jsonl_file}")
        return

    # Initialize image editor with provider and model
    try:
        editor = ImagePromptEditor(vlm_provider=provider)
        print(f"✓ Initialized {provider} VLM provider with model {model}")
    except Exception as e:
        print(f"Error initializing VLM provider: {e}")
        return

    

    # Process the images
    process_jsonl_images(jsonl_file, str(output_directory), editor=editor, max_workers=max_threads)


if __name__ == "__main__":
    main()