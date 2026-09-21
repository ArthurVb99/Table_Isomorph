"""Edit table images and project the edited result to validated JSON."""

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

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

		# Initialize provider settings once; both parents otherwise do this.
		ImageProjectionProcessor.__init__(self, vlm_provider=vlm_provider, model=model)

		template_dir = Path(__file__).parent.parent / "templates" / "image_edit_and_project_prompts"
		self.jinja_env.loader.searchpath = [str(template_dir)]

	def _create_edit_and_project_prompt(self, image_path: str) -> str:
		"""Render the prompt for transposing and extracting a table image."""
		with Image.open(image_path) as image:
			target_width, target_height = image.height, image.width

		return self.render_template(
			"table_transpose_and_structure_extraction.txt",
			target_w=target_width,
			target_h=target_height,
			target_width=target_width,
			target_height=target_height,
			target_ratio=target_width / target_height if target_height else 1.0,
		)

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
				return None, f"Validation error: {error}"

			if model:
				model.imgid = imgid
				model.split = split
				model.filename = Path(image_path).name

			return model, None
		except Exception as error:
			return None, f"Processing error: {error}"

	def project_edited_table(self, image_path: str, imgid: int = 1, split: str = "train") -> Optional[dict]:
		"""Return the edited table projection as a JSON-serializable dictionary."""
		model, error = self.process_edit_and_project(image_path, imgid, split)
		if error:
			raise ValueError(error)
		return model.model_dump() if model else None
