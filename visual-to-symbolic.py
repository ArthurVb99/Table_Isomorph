import os
from pathlib import Path

from batch_image_processor import MAX_IMG
from batch_structure_extraction import find_images_in_directory, process_images_to_jsonl
from track_4_image_edit_and_project.image_editor_projection import ImagePromptEditorProjection

def main():
    """Main function - configure paths here"""

    # Get configuration from environment variables
    input_dir = os.getenv("PATH_INPUT_IMAGES", "")
    output_jsonl = os.getenv("PATH_OUTPUT_JSONL", "extracted_structures.jsonl")
    vlm_provider = os.getenv("VLM_PROVIDER", "ollama")
    model = os.getenv("MODEL_PROJECTION", "qwen3.8")
    max_threads = int(os.getenv("MAX_THREADS", "4"))

    # format the output file witht vlm provider and model
    if not input_dir:
        print("Error: PATH_INPUT_IMAGES environment variable not set")
        print("Set it to the directory containing images or a single image path")
        return

    output_jsonl = Path(input_dir).parent / output_jsonl.replace(
        '.jsonl', f'#{vlm_provider}_{model}#.jsonl'
    )

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

    max_images = int(MAX_IMG)
    image_paths = image_paths[:max_images]

    print(f"Found {len(image_paths)} images to process")

    # Initialize processor
    try:
        processor = ImagePromptEditorProjection(
            vlm_provider=vlm_provider,
            model=model,
        )
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