import json
import os

from track_1_llm_prompt.llm_processor import LLMPromptProcessor


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