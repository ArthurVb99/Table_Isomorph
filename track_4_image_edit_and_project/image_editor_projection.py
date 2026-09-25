"""Edit table images and project the edited result to validated JSON."""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

from track_2_image_editing.image_editor import ImagePromptEditor
from track_3_image_projection.image_projection import ImageProjectionProcessor
from track_1_llm_prompt.validators import LLMTableModel, validate_llm_output


class ImagePromptEditorProjection(ImageProjectionProcessor, ImagePromptEditor):
	"""Transpose a table image and extract its structure as validated JSON.

	The class combines the image editing helpers from ``ImagePromptEditor``
	with the structured VLM processing from ``ImageProjectionProcessor``.
	Combined-operation prompts are loaded from ``image_edit_and_project_prompts``.
	"""

	def __init__(
		self,
		output_dir: str = "output_images",
		vlm_provider: str = "ollama",
		model: str = "qwen3.8",
	):
		"""Initialize the shared VLM client and combined-operation templates."""
		self.output_dir = output_dir
		Path(self.output_dir).mkdir(parents=True, exist_ok=True)
		self._last_vlm_metadata = {}

		# Initialize provider settings once; both parents otherwise do this.
		ImageProjectionProcessor.__init__(self, vlm_provider=vlm_provider, model=model)

		template_dir = Path(__file__).parent.parent / "templates" / "image_edit_and_project_prompts"
		self.jinja_env.loader.searchpath = [str(template_dir)]

	def _create_edit_and_project_prompt(self, image_path: str) -> str:
		"""Render the prompt for transposing and extracting a table image."""
		return self.render_template("table_transpose_and_structure_extraction.txt")

	def process_edit_and_project(
		self,
		image_path: str,
		imgid: int = 1,
		split: str = "train",
	) -> Tuple[Optional[LLMTableModel], Optional[str]]:
		"""Transpose and extract a table image using the combined prompt.

		Returns the validated table model and an error message, matching the
		return contract of ``ImageProjectionProcessor.extract_table_structure``.
		"""
		try:
			prompt = self._create_edit_and_project_prompt(image_path)
			json_response = self._process_with_vlm(image_path, prompt)

			if not json_response:
				return None, "No response from VLM"

			model, error = validate_llm_output(json_response)
			if error:
				artifact_path = self._save_invalid_response(image_path, json_response, error)
				return None, f"Validation error: {error} (raw response: {artifact_path})"

			if model:
				model.imgid = imgid
				model.split = split
				model.filename = Path(image_path).name

			return model, None
		except Exception as error:
			return None, f"Processing error: {error}"

	def _save_invalid_response(self, image_path: str, response: object, error: str) -> str:
		"""Save an invalid model response and its validation error for inspection."""
		output_dir = Path(os.getenv("INVALID_OUTPUT_DIR", "invalid_outputs"))
		if output_dir.name != self.model:
			output_dir /= self.model
		output_dir.mkdir(parents=True, exist_ok=True)
		stem = Path(image_path).stem
		response_path = output_dir / f"{stem}.response.txt"
		error_path = output_dir / f"{stem}.error.txt"

		if isinstance(response, str):
			response_text = response
		else:
			response_text = json.dumps(response, ensure_ascii=False, indent=2)
		response_path.write_text(response_text, encoding="utf-8")
		error_details = {
			"error": error,
			"done": self._last_vlm_metadata.get("done"),
			"done_reason": self._last_vlm_metadata.get("done_reason"),
			"prompt_eval_count": self._last_vlm_metadata.get("prompt_eval_count"),
			"eval_count": self._last_vlm_metadata.get("eval_count"),
		}
		error_path.write_text(json.dumps(error_details, indent=2), encoding="utf-8")
		return str(response_path)

	def extract_table_structure(
		self,
		image_path: str,
		imgid: int = 1,
		split: str = "train",
	) -> Tuple[Optional[LLMTableModel], Optional[str]]:
		"""Use the combined edit-and-project flow for batch extractor compatibility."""
		return self.process_edit_and_project(image_path, imgid, split)

	def project_edited_table(self, image_path: str, imgid: int = 1, split: str = "train") -> Optional[dict]:
		"""Return the edited table projection as a JSON-serializable dictionary."""
		model, error = self.process_edit_and_project(image_path, imgid, split)
		if error:
			raise ValueError(error)
		return model.model_dump() if model else None
