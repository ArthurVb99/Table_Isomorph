import os
import json
from pathlib import Path

from batch_image_processor import MAX_IMG
from batch_process_jsonl import process_jsonl_data
from batch_structure_extraction import process_images_to_jsonl
from evaluate_metrics import TableMetricsEvaluator
from track_1_llm_prompt.llm_processor import LLMPromptProcessor
from track_4_image_edit_and_project.image_editor_projection import ImagePromptEditorProjection


PUBTABNET_JSONL = os.getenv(
    "PUBTABNET_JSONL",
    "/data/brussel/vo/000/bvo00018/vsc11306/cross-modal_experiments/PubTabNet/PubTabNet_2.0.0.jsonl",
)


def select_tables(dataset_path: str, split: str, max_images: int, max_tokens: int,
                  model: str) -> tuple[dict, set]:
    """Select the first image records within the input-token limit."""
    token_counter = LLMPromptProcessor(provider="ollama", model=model)
    image_paths = {}

    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if len(image_paths) >= max_images:
                break
            data = json.loads(line)
            if data.get("split") != split:
                continue
            if token_counter.count_input_tokens(json.dumps(data)) > max_tokens:
                continue
            image_paths[data["imgid"]] = data["filename"]

    return image_paths, set(image_paths)


def main():
    """Main function - configure paths here"""

    # Get configuration from environment variables
    input_dir = os.getenv("PATH_INPUT_IMAGES", "")
    output_jsonl = os.getenv("PATH_OUTPUT_JSONL", "extracted_structures.jsonl")
    vlm_provider = os.getenv("VLM_PROVIDER", "ollama")
    model = os.getenv("MODEL_PROJECTION", "qwen3.8")
    max_tokens = int(os.getenv("MAX_TOKENS", "5000"))
    max_threads = int(os.getenv("MAX_THREADS", "4"))
    split_focus = os.getenv("SPLIT_FOCUS", "train")

    # format the output file witht vlm provider and model
    if not input_dir:
        print("Error: PATH_INPUT_IMAGES environment variable not set")
        print("Set it to the directory containing images or a single image path")
        return

    output_jsonl = Path(input_dir).parent / output_jsonl.replace(
        '.jsonl', f'#{vlm_provider}_{model}#.jsonl'
    )
    os.environ.setdefault(
        "INVALID_OUTPUT_DIR",
        str(output_jsonl.parent / "invalid_outputs" / model),
    )

    if not Path(input_dir).is_dir():
        print(f"Error: Input image directory does not exist: {input_dir}")
        return

    if not Path(PUBTABNET_JSONL).is_file():
        print(f"Error: Dataset JSONL not found: {PUBTABNET_JSONL}")
        return

    image_paths_by_id, selected_imgids = select_tables(
        PUBTABNET_JSONL,
        split_focus,
        int(MAX_IMG),
        max_tokens,
        model,
    )
    selected_images = [
        (imgid, Path(input_dir) / filename)
        for imgid, filename in image_paths_by_id.items()
        if (Path(input_dir) / filename).is_file()
    ]
    selected_image_ids = [imgid for imgid, _ in selected_images]
    selected_imgids = set(selected_image_ids)
    image_paths = [str(path) for _, path in selected_images]

    if not image_paths:
        print(f"No selected images found in {input_dir}")
        return

    print(f"Selected {len(selected_imgids)} tables under {max_tokens} input tokens")
    print(f"Found {len(image_paths)} selected images to process")

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
        max_workers=max_threads,
        image_ids=selected_image_ids,
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

    batch_processor = LLMPromptProcessor(
        provider=os.getenv("LLM_PROVIDER", "ollama"),
        model=os.getenv("MODEL_LLM", model),
    )
    batch_output = Path(PUBTABNET_JSONL).with_name(
        f"{Path(PUBTABNET_JSONL).stem}_{split_focus}_processed_{batch_processor.model}.jsonl"
    )
    print(f"Batch processing JSONL data from {PUBTABNET_JSONL} to {batch_output}")
    process_jsonl_data(
        PUBTABNET_JSONL,
        str(batch_output),
        batch_processor,
        split_focus=split_focus,
        max_tokens=max_tokens,
        max_workers=max_threads,
        selected_imgids=selected_imgids,
    )

    metrics_output = Path(os.getenv(
        "PATH_OUTPUT_RESULTS",
        str(batch_output.with_name(f"{batch_output.stem}_metrics.jsonl")),
    ))
    evaluator = TableMetricsEvaluator(
        structure_only=bool(int(os.getenv("STRUCTURE_ONLY", "0"))),
        ignored_nodes=[],
    )
    print(f"Evaluating results from {output_jsonl} against {batch_output}")
    evaluation = evaluator.evaluate_jsonl_files(
        pred_jsonl=str(output_jsonl),
        gt_jsonl=batch_output,
        output_path=str(metrics_output),
        max_workers=max_threads,
        matching_by_imgid=bool(int(os.getenv("MATCHING_BY_IMGID", "1"))),
    )
    print(f"Evaluation metrics saved to: {metrics_output}")
    print(f"Matched entries: {evaluation['metrics']['matched']}")


if __name__ == "__main__":
    main()