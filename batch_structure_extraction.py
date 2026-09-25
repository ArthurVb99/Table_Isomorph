"""
Batch Table Structure Extraction Script for Track 3
Reads images, processes them using ImageProjectionProcessor, and saves extracted JSON structures.
Uses multi-threading for parallel processing.
"""

import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from track_3_image_projection.image_projection import ImageProjectionProcessor


def process_single_image_structure(item_data: dict, output_path: Path, processor: ImageProjectionProcessor, lock: Lock) -> tuple:
    """
    Process a single image to extract table structure.

    Args:
        item_data (dict): Data containing image path and metadata
        output_path (Path): Output directory path
        processor (ImageProjectionProcessor): Image projection processor instance
        lock (Lock): Thread lock for safe printing

    Returns:
        tuple: (success: bool, image_path: str, result: dict or None, error: str or None)
    """
    image_path = item_data['image_path']
    imgid = item_data.get('imgid', 1)
    split = item_data.get('split', 'train')

    try:
        with lock:
            print(f"Processing: {Path(image_path).name} (imgid: {imgid})")

        # Extract table structure
        model, error = processor.extract_table_structure(image_path, imgid, split)

        if model:
            result = model.model_dump()
            with lock:
                print(f"✓ Successfully extracted structure for {Path(image_path).name}")
            return True, image_path, result, None
        else:
            with lock:
                print(f"✗ Failed to extract structure for {Path(image_path).name}: {error}")
            return False, image_path, None, error

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        with lock:
            print(f"✗ Error processing {Path(image_path).name}: {error_msg}")
        return False, image_path, None, error_msg


def process_images_to_jsonl(image_paths: list, output_jsonl_path: str,
                           processor: ImageProjectionProcessor = None, max_workers: int = 4,
                           image_ids: list = None) -> dict:
    """
    Process multiple images to extract table structures and save to JSONL.

    Args:
        image_paths (list): List of image paths to process
        output_jsonl_path (str): Path to output JSONL file
        processor (ImageProjectionProcessor): Processor instance (created if None)
        max_workers (int): Maximum number of concurrent threads
        image_ids (list): Optional dataset imgids aligned with image_paths

    Returns:
        dict: Processing results and statistics
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_jsonl_path).parent
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize processor if not provided
    if processor is None:
        processor = ImageProjectionProcessor()

    # Thread lock for safe printing
    print_lock = Lock()

    # Prepare items to process
    items_to_process = []
    for i, image_path in enumerate(image_paths, 1):
        # Extract metadata from path
        path_obj = Path(image_path)
        split = "train"  # Default
        imgid = image_ids[i - 1] if image_ids is not None else i

        items_to_process.append({
            'image_path': image_path,
            'imgid': imgid,
            'split': split
        })

    print(f"Processing {len(items_to_process)} images with {max_workers} threads")
    print(f"Output will be saved to: {output_jsonl_path}")
    print(f"Invalid responses will be saved to: {output_path / 'invalid_outputs'}")
    print("=" * 60)

    # Process images in parallel
    results = []
    stats = {"processed": 0, "successful": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_single_image_structure, item, output_path, processor, print_lock): item
            for item in items_to_process
        }

        # Collect results as they complete
        for future in as_completed(future_to_item):
            success, image_path, result, error = future.result()
            stats["processed"] += 1

            if success and result:
                results.append(result)
                stats["successful"] += 1
            else:
                stats["failed"] += 1

    # Save results to JSONL
    if results:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\n✓ Saved {len(results)} extracted structures to {output_jsonl_path}")
    else:
        print("\n⚠ No valid structures extracted")

    return {"results": results, "statistics": stats}


def find_images_in_directory(base_dir: str, extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp')) -> list:
    """
    Recursively find all images in a directory.

    Args:
        base_dir (str): Base directory to search
        extensions (tuple): File extensions to include

    Returns:
        list: List of image paths
    """
    base_path = Path(base_dir)
    image_paths = []

    if base_path.is_file():
        if base_path.suffix.lower() in extensions:
            return [str(base_path)]
        return []

    for ext in extensions:
        image_paths.extend([str(p) for p in base_path.rglob(f"*{ext}")])

    return sorted(image_paths)


def main():
    """Main function - configure paths here"""

    # Get configuration from environment variables
    input_dir = os.getenv("PATH_INPUT_IMAGES", "")
    output_jsonl = os.getenv("PATH_OUTPUT_JSONL", "extracted_structures.jsonl")
    vlm_provider = os.getenv("VLM_PROVIDER", "openai")
    model = os.getenv("MODEL_PROJECTION", "gpt-5")
    max_threads = int(os.getenv("MAX_THREADS", "4"))

    # format the output file witht vlm provider and model
    output_jsonl = Path(input_dir).parent / output_jsonl.replace('.jsonl', f'#{vlm_provider}_{model}#.jsonl')
    
    if not input_dir:
        print("Error: PATH_INPUT_IMAGES environment variable not set")
        print("Set it to the directory containing images or a single image path")
        return

    # Check if input is a single file or directory
    input_path = Path(input_dir)
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        image_paths = find_images_in_directory(input_dir)
    else:
        print(f"Error: Input path {input_dir} does not exist")
        return

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process")

    # Initialize processor
    try:
        processor = ImageProjectionProcessor(vlm_provider=vlm_provider, model=model)
        print(f"✓ Initialized {vlm_provider} VLM provider with model {model}")
    except Exception as e:
        print(f"Error initializing VLM provider: {e}")
        return

    # Process images
    result = process_images_to_jsonl(
        image_paths=image_paths,
        output_jsonl_path=output_jsonl,
        processor=processor,
        max_workers=max_threads
    )

    # Print final statistics
    stats = result["statistics"]
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images processed: {stats['processed']}")
    print(f"Successful extractions: {stats['successful']}")
    print(f"Failed extractions: {stats['failed']}")
    print(f"Success rate: {(stats['successful']/stats['processed']*100):.1f}%" if stats['processed'] > 0 else "0%")
    print(f"Output file: {output_jsonl}")


if __name__ == "__main__":
    main()