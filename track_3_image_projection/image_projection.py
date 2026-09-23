"""
Image Projection Processor for Track 3
Processes images to extract table structures and return JSON representations.
Similar to ImagePromptEditor but returns structured JSON data instead of edited images.
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
import requests

from track_1_llm_prompt.validators import validate_llm_output, LLMTableModel

load_dotenv()


class ImageProjectionProcessor:
    """
    Process images to extract table structures and return JSON representations.
    Uses VLM providers to analyze table images and generate structured data.
    """

    def __init__(self, vlm_provider: str = "openai", model: str = "gpt-4-vision-preview"):
        """
        Initialize the Image Projection Processor.

        Args:
            vlm_provider (str): VLM provider ('openai', 'claude', 'openrouter', 'huggingface', 'ollama')
            model (str): Specific model to use
        """
        self.vlm_provider = vlm_provider.lower()
        self.model = model

        # Initialize VLM client based on provider
        self._initialize_vlm()

        # Setup Jinja2 template loader for image prompts
        template_dir = Path(__file__).parent.parent / "templates" / "image_prompts"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

    def _initialize_vlm(self):
        """Initialize VLM client based on provider."""
        if self.vlm_provider == "openai":
            self._init_openai_vlm()
        elif self.vlm_provider == "claude":
            self._init_claude_vlm()
        elif self.vlm_provider == "openrouter":
            self._init_openrouter_vlm()
        elif self.vlm_provider == "huggingface":
            self._init_huggingface_vlm()
        elif self.vlm_provider == "ollama":
            self._init_ollama_vlm()
        else:
            raise ValueError(f"Unsupported VLM provider: {self.vlm_provider}")

    def _init_openai_vlm(self):
        """Initialize OpenAI VLM provider."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def _init_claude_vlm(self):
        """Initialize Claude VLM provider."""
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.claude_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    def _init_openrouter_vlm(self):
        """Initialize OpenRouter VLM provider."""
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://api.openrouter.ai/v1")

    def _init_huggingface_vlm(self):
        """Initialize HuggingFace VLM provider."""
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.hf_api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
        self.hf_base_url = os.getenv("HUGGINGFACE_BASE_URL", "https://api-inference.huggingface.co/models")

    def _init_ollama_vlm(self):
        """Initialize Ollama VLM provider."""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def load_template(self, template_name: str) -> str:
        """
        Load an image prompt template from file.

        Args:
            template_name (str): Name of template file (e.g., 'table_structure_extraction.txt')

        Returns:
            str: Template content
        """
        try:
            template = self.jinja_env.get_template(template_name)
            return template
        except Exception as e:
            raise FileNotFoundError(f"Template not found: {template_name}. Error: {str(e)}")

    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render a template with provided variables.

        Args:
            template_name (str): Template file name
            **kwargs: Variables to populate in template

        Returns:
            str: Rendered prompt text
        """
        template = self.load_template(template_name)
        return template.render(**kwargs)

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 string.

        Args:
            image_path (str): Path to image file

        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_table_structure(self, image_path: str, imgid: int = 1, split: str = "train") -> Tuple[Optional[LLMTableModel], Optional[str]]:
        """
        Extract table structure from image and return validated JSON.

        Args:
            image_path (str): Path to table image
            imgid (int): Image ID for the structure
            split (str): Data split (train/val/test)

        Returns:
            Tuple[Optional[LLMTableModel], Optional[str]]: (validated_model, error_message)
        """
        try:
            # Create prompt for table structure extraction
            prompt = self._create_table_extraction_prompt()

            # Process with VLM
            json_response = self._process_with_vlm(image_path, prompt)

            if not json_response:
                return None, "No response from VLM"

            # Parse and validate the response
            model, error = validate_llm_output(json_response)
            if error:
                return None, f"Validation error: {error}"

            # Update model with provided metadata
            if model:
                model.imgid = imgid
                model.split = split
                model.filename = Path(image_path).name

            return model, None

        except Exception as e:
            return None, f"Processing error: {str(e)}"

    def _create_table_extraction_prompt(self) -> str:
        """
        Create the prompt for table structure extraction using template.

        Returns:
            str: Formatted prompt for VLM
        """
        return self.render_template("table_structure_extraction.txt")

    def _process_with_vlm(self, image_path: str, prompt: str) -> Optional[str]:
        """
        Process image with VLM and return JSON response.

        Args:
            image_path (str): Path to image
            prompt (str): Text prompt

        Returns:
            str: JSON response from VLM
        """
        try:
            if self.vlm_provider == "openai":
                return self._process_openai_vlm(image_path, prompt)
            elif self.vlm_provider == "claude":
                return self._process_claude_vlm(image_path, prompt)
            elif self.vlm_provider == "openrouter":
                return self._process_openrouter_vlm(image_path, prompt)
            elif self.vlm_provider == "huggingface":
                return self._process_huggingface_vlm(image_path, prompt)
            elif self.vlm_provider == "ollama":
                return self._process_ollama_vlm(image_path, prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.vlm_provider}")
        except Exception as error:
            raise RuntimeError(f"VLM processing error: {error}") from error
    def enforce_no_additional_properties(self, schema: dict) -> dict:
        """
        Recursively set additionalProperties=False on every object schema.
        Required by Structured Outputs when strict=True.
        """
        if isinstance(schema, dict):
            # If this node is an object schema, enforce additionalProperties: false
            if schema.get("type") == "object":
                schema["additionalProperties"] = False

            # Recurse into common schema containers
            for k, v in list(schema.items()):
                if isinstance(v, (dict, list)):
                    schema[k] = self.enforce_no_additional_properties(v)

        elif isinstance(schema, list):
            return [self.enforce_no_additional_properties(x) for x in schema]

        return schema


    def _process_openai_vlm(self, image_path: str, prompt: str) -> Optional[str]:
        """Process with OpenAI Vision."""
        base64_image = self.encode_image_to_base64(image_path)

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        schema = LLMTableModel.model_json_schema()  # Pydantic v2
        schema = self.enforce_no_additional_properties(schema)
        payload = {
            "model": "gpt-5", 
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"},
                ],
            }],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "table_tsr",
                    "strict": True,
                    "schema": schema,
                }
            },
        }

        # payload = {
        #     "model": "gpt-5",
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "input_text", "text": prompt},
        #                 {
        #                     "type": "input_image",
        #                     "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        #                 }
        #             ]
        #         }
        #     ],
        #     #"max_tokens": 6000,
        #     #"temperature": 1  # Low temperature for structured output
        # }

        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            #timeout=120
        )
    
        response.raise_for_status()

        result = response.json()
        # Robust extraction for /v1/responses
        text_out = None
        for item in result.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") in ("output_text", "text"):
                        text_out = c.get("text")
                        break
            if text_out:
                break

        if not text_out:
            raise RuntimeError(f"No text output found. Full response: {result}")

        return text_out

    def _process_claude_vlm(self, image_path: str, prompt: str) -> Optional[str]:
        """Process with Claude Vision."""
        base64_image = self.encode_image_to_base64(image_path)

        headers = {
            "x-api-key": self.claude_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        data = {
            "model": self.model,
            "max_tokens": 4000,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return result["content"][0]["text"]

    def _process_openrouter_vlm(self, image_path: str, prompt: str) -> Optional[str]:
        """Process with OpenRouter."""
        base64_image = self.encode_image_to_base64(image_path)

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", ""),
            "X-Title": os.getenv("OPENROUTER_TITLE", "")
        }

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.1
        }

        response = requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _process_huggingface_vlm(self, image_path: str, prompt: str) -> Optional[str]:
        """Process with HuggingFace (may not support vision models well)."""
        # Note: HuggingFace may not have good vision models for this task
        # This is a placeholder implementation
        base64_image = self.encode_image_to_base64(image_path)

        headers = {"Authorization": f"Bearer {self.hf_api_key}"}

        # Use a generic model - may need adjustment based on available models
        model_id = "microsoft/DialoGPT-medium"  # Placeholder - vision models may not be available

        data = {
            "inputs": prompt,  # Text-only for now
            "parameters": {"max_length": 1000, "temperature": 0.1}
        }

        response = requests.post(
            f"{self.hf_base_url}/{model_id}",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "")
        return str(result)

    def _process_ollama_vlm(self, image_path: str, prompt: str) -> Optional[str]:
        """Process with Ollama (using vision-capable models like llava)."""
        base64_image = self.encode_image_to_base64(image_path)

        request_timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))

        data = {
            "model": self.model,  # e.g., "llava", "bakllava"
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "temperature": 0.1,
            "format": "json",
            "think": False,
            "options": {"num_predict": num_predict},
        }

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=data,
                timeout=request_timeout
            )
            # result = response.json()
            # print({
            #     "done": result.get("done"),
            #     "done_reason": result.get("done_reason"),
            #     "prompt_eval_count": result.get("prompt_eval_count"),
            #     "eval_count": result.get("eval_count"),
            # })
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.ConnectionError as error:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.ollama_base_url}: {error}"
            ) from error
        except requests.exceptions.HTTPError as error:
            response_text = error.response.text.strip() if error.response is not None else ""
            detail = f": {response_text}" if response_text else ""
            raise RuntimeError(
                f"Ollama request failed for model '{self.model}' "
                f"with HTTP {error.response.status_code if error.response is not None else 'error'}{detail}"
            ) from error

    def batch_process_images(self, image_paths: list, output_jsonl_path: str = None) -> Dict[str, Any]:
        """
        Process multiple images and optionally save results to JSONL.

        Args:
            image_paths (list): List of image paths to process
            output_jsonl_path (str): Optional path to save results as JSONL

        Returns:
            Dict: Processing results and statistics
        """
        results = []
        stats = {"processed": 0, "successful": 0, "failed": 0}

        for i, image_path in enumerate(image_paths, 1):
            print(f"Processing image {i}/{len(image_paths)}: {image_path}")

            # Extract filename info for imgid and split
            path_obj = Path(image_path)
            imgid = i  # Simple incremental ID
            split = "train"  # Default split

            # Try to extract split from path
            if "train" in str(path_obj):
                split = "train"
            elif "val" in str(path_obj) or "valid" in str(path_obj):
                split = "val"
            elif "test" in str(path_obj):
                split = "test"

            model, error = self.extract_table_structure(image_path, imgid, split)
            stats["processed"] += 1

            if model:
                result = model.model_dump()
                results.append(result)
                stats["successful"] += 1
                print(f"✓ Successfully extracted structure for {path_obj.name}")
            else:
                stats["failed"] += 1
                print(f"✗ Failed to extract structure: {error}")

        # Save to JSONL if requested
        if output_jsonl_path and results:
            with open(output_jsonl_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            print(f"✓ Saved {len(results)} results to {output_jsonl_path}")

        return {"results": results, "statistics": stats}


if __name__ == "__main__":
    # Example usage
    processor = ImageProjectionProcessor(vlm_provider="openai", model="gpt-4-vision-preview")

    # Test with single image
    test_image = "path/to/table/image.png"
    if os.path.exists(test_image):
        model, error = processor.extract_table_structure(test_image)
        if model:
            print("Extracted structure:", json.dumps(model.model_dump(), indent=2))
        else:
            print("Error:", error)
    else:
        print("Test image not found. Please provide a valid image path.")
