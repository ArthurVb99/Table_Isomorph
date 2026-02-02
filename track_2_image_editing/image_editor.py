"""
Image Editor for Track 2
Processes images based on text prompts and generates edited output images.
Supports template-based prompt processing and VLM (Vision-Language Model) integration.
"""

import os
import base64
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from typing import Optional
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from dotenv import load_dotenv
import requests

load_dotenv()


class ImagePromptEditor:
    """
    Edit and manipulate images based on text prompts or templates.
    Supports various image processing operations and VLM integration for advanced tasks.
    Supports multiple VLM providers: OpenAI, Claude, OpenRouter, HuggingFace, Ollama
    """

    def __init__(self, output_dir: str = "output_images", vlm_provider: str = "openai"):
        """
        Initialize the Image Editor.

        Args:
            output_dir (str): Directory to save output images
            vlm_provider (str): VLM provider ('openai', 'claude', 'openrouter', 'huggingface', 'ollama')
        """
        self.output_dir = output_dir
        self.vlm_provider = vlm_provider.lower()
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize VLM client based on provider
        self._initialize_vlm()

        # Setup Jinja2 template loader
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
            template_name (str): Name of template file (e.g., 'basic_edits.txt')

        Returns:
            str: Template content
        """
        try:
            template = self.jinja_env.get_template(template_name)
            return template
        except TemplateNotFound  as e:
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

    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from file.

        Args:
            image_path (str): Path to the image file

        Returns:
            Image.Image: PIL Image object
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path)

    def adjust_brightness(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """
        Adjust image brightness.

        Args:
            image (Image.Image): Input image
            factor (float): Brightness factor (1.0 = no change, >1.0 = brighter)

        Returns:
            Image.Image: Edited image
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def adjust_contrast(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """
        Adjust image contrast.

        Args:
            image (Image.Image): Input image
            factor (float): Contrast factor (1.0 = no change, >1.0 = higher contrast)

        Returns:
            Image.Image: Edited image
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def apply_blur(self, image: Image.Image, radius: int = 2) -> Image.Image:
        """
        Apply blur filter to image.

        Args:
            image (Image.Image): Input image
            radius (int): Blur radius

        Returns:
            Image.Image: Blurred image
        """
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def apply_sharpen(self, image: Image.Image, factor: float = 2.0) -> Image.Image:
        """
        Sharpen the image.

        Args:
            image (Image.Image): Input image
            factor (float): Sharpness factor (1.0 = no change, >1.0 = sharper)

        Returns:
            Image.Image: Sharpened image
        """
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    def apply_grayscale(self, image: Image.Image) -> Image.Image:
        """
        Convert image to grayscale.

        Args:
            image (Image.Image): Input image

        Returns:
            Image.Image: Grayscale image
        """
        return image.convert("L")

    def apply_saturation(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        """
        Adjust color saturation.

        Args:
            image (Image.Image): Input image
            factor (float): Saturation factor (1.0 = no change, >1.0 = more saturated)

        Returns:
            Image.Image: Edited image
        """
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def resize_image(self, image: Image.Image, width: int = None, height: int = None) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.

        Args:
            image (Image.Image): Input image
            width (int): Target width
            height (int): Target height

        Returns:
            Image.Image: Resized image
        """
        if width and height:
            return image.resize((width, height), Image.Resampling.LANCZOS)
        elif width:
            ratio = width / image.width
            height = int(image.height * ratio)
            return image.resize((width, height), Image.Resampling.LANCZOS)
        elif height:
            ratio = height / image.height
            width = int(image.width * ratio)
            return image.resize((width, height), Image.Resampling.LANCZOS)
        return image

    def save_image(self, image: Image.Image, filename: str) -> str:
        """
        Save edited image to output directory.

        Args:
            image (Image.Image): Image to save
            filename (str): Output filename

        Returns:
            str: Path to saved image
        """
        output_path = os.path.join(self.output_dir, filename)
        image.save(output_path)
        return output_path

    def process_with_prompt(self, image_path: str, prompt: str, output_filename: str = "output.png") -> str:
        """
        Process image based on text prompt instructions.

        Args:
            image_path (str): Path to input image
            prompt (str): Text prompt describing desired edits
            output_filename (str): Output filename

        Returns:
            str: Path to output edited image
        """
        image = self.load_image(image_path)

        # Parse prompt for common editing operations
        prompt_lower = prompt.lower()
        
        if "brighten" in prompt_lower or "bright" in prompt_lower:
            image = self.adjust_brightness(image, 1.3)
        
        if "darken" in prompt_lower or "dark" in prompt_lower:
            image = self.adjust_brightness(image, 0.7)
        
        if "sharpen" in prompt_lower or "sharp" in prompt_lower:
            image = self.apply_sharpen(image, 2.0)
        
        if "blur" in prompt_lower or "blurred" in prompt_lower:
            image = self.apply_blur(image, 3)
        
        if "grayscale" in prompt_lower or "gray" in prompt_lower or "bw" in prompt_lower:
            image = self.apply_grayscale(image)
        
        if "contrast" in prompt_lower:
            image = self.adjust_contrast(image, 1.3)
        
        if "saturate" in prompt_lower or "vivid" in prompt_lower:
            image = self.apply_saturation(image, 1.5)
        
        if "desaturate" in prompt_lower or "dull" in prompt_lower:
            image = self.apply_saturation(image, 0.5)

        # Handle table transposition (placeholder for future implementation)
        if "transpose" in prompt_lower and "table" in prompt_lower:
            print("Transposing table in image (placeholder implementation)")
            image = self.transpose_table_image(image)

        return self.save_image(image, output_filename)

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 for VLM processing.

        Args:
            image_path (str): Path to image file

        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def decode_image_from_base64(self, base64encode_str):
        """
        Decode base64 string back to PIL Image.

        Args:
            base64encode_str (str): Base64 encoded image string

        Returns:
            Image.Image: Decoded PIL Image object
        """
        import io
        decoded = base64.b64decode(base64encode_str)
        image = Image.open(io.BytesIO(decoded))
        return image

    def process_with_vlm(self, image_path: str, prompt: str, output_filename: str = "vlm_output.png") -> str:
        """
        Process image using a Vision-Language Model (VLM).

        Args:
            image_path (str): Path to input image
            prompt (str): Text prompt for VLM
            output_filename (str): Output filename

        Returns:
            str: Path to output image (downloaded from VLM)
        """
        try:
            if self.vlm_provider == "openai":
                return self._process_openai_vlm(image_path, prompt, output_filename)
            elif self.vlm_provider == "claude":
                return self._process_claude_vlm(image_path, prompt, output_filename)
            elif self.vlm_provider == "openrouter":
                return self._process_openrouter_vlm(image_path, prompt, output_filename)
            elif self.vlm_provider == "huggingface":
                return self._process_huggingface_vlm(image_path, prompt, output_filename)
            elif self.vlm_provider == "ollama":
                return self._process_ollama_vlm(image_path, prompt, output_filename)
            else:
                raise ValueError(f"Unsupported VLM provider: {self.vlm_provider}")
        except Exception as e:
            print(f"VLM processing failed: {e}")
            # Fallback to traditional image processing
            # return self.process_with_prompt(image_path, prompt, output_filename)

    def _process_openai_vlm(self, image_path: str, prompt: str, output_filename: str) -> str:
        """
        Edit an input image using a text prompt, and save the edited image to output_filename.

        Uses:
        - POST https://api.openai.com/v1/responses
        - tool: image_generation
        - input: [input_text + input_image]
        - output: base64-encoded image in an image_generation_call result

        Ref: Responses API + image_generation tool returns a base64 image result. :contentReference[oaicite:1]{index=1}
        """
        base64_image = self.encode_image_to_base64(image_path)

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            # Any model that supports tool use; the tool itself uses GPT Image models underneath. :contentReference[oaicite:2]{index=2}
            "model": "gpt-5",
            
            "tools": [{"type": "image_generation"}],
            # Force an image output (so you reliably get an edited image back)
            "tool_choice": {"type": "image_generation"},
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ],
                }
            ],
            # Optional knobs supported by the tool (keep or remove as you like)
            "metadata": {"purpose": "image_edit"},
        }

        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            #timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()

        # Extract the base64 image from the tool call output
        # The image_generation_call includes a base64-encoded image in "result". :contentReference[oaicite:3]{index=3}
        image_b64: Optional[str] = None
        for item in result.get("output", []):
            if item.get("type") == "image_generation_call" and item.get("result"):
                image_b64 = item["result"]
                break

        if not image_b64:
            # Helpful debug if the API returns something unexpected
            raise RuntimeError(f"No image_generation_call result found. Full response: {result}")

        # Save edited image
        with open(output_filename, "wb") as f:
            f.write(base64.b64decode(image_b64))

        return output_filename

    # def _process_openai_vlm(self, image_path: str, prompt: str, output_filename: str) -> str:
    #     """
    #     Generates an image using OpenAI's Image Generation endpoint.

    #     What this does:
    #     - Uses /v1/images/generations (prompt -> image) to produce an output image.
    #     - Saves the returned image (base64) to `output_filename` and returns its path.

    #     Important:
    #     - The image generation endpoint does NOT take an input image for "vision analysis".
    #     - If you want image+prompt -> edited image, use the Responses API with the
    #         image_generation tool (image editing), not /v1/images/generations.
    #     """
    #     headers = {
    #         "Authorization": f"Bearer {self.openai_api_key}",
    #         "Content-Type": "application/json",
    #     }

    #     payload = {
    #         "model": "gpt-image-1",          # use an image generation model
    #         "prompt": prompt,                # prompt -> image
    #         "size": "1024x1024",
    #         "output_format": "png",
    #         # Optional: "quality": "high",
    #         # Optional: "background": "transparent",
    #     }

    #     resp = requests.post(
    #         "https://api.openai.com/v1/images/generations",
    #         headers=headers,
    #         json=payload,
    #         # timeout=120,
    #     )
    #     resp.raise_for_status()

    #     result = resp.json()

    #     # Images API returns base64 in data[0]["b64_json"]
    #     b64 = result["data"][0]["b64_json"]
    #     img_bytes = base64.b64decode(b64)

    #     # Save to the requested output filename
    #     with open(output_filename, "wb") as f:
    #         f.write(img_bytes)

    #     print("Image generation raw response:", result)
    #     return output_filename

    # def _process_openai_vlm_0(self, image_path: str, prompt: str, output_filename: str) -> str:
    #     """
    #     Process image using OpenAI GPT-4V.

    #     Args:
    #         image_path (str): Path to input image
    #         prompt (str): Text prompt
    #         output_filename (str): Output filename

    #     Returns:
    #         str: Path to output image
    #     """
    #     base64_image = self.encode_image_to_base64(image_path)

    #     headers = {
    #         "Authorization": f"Bearer {self.openai_api_key}",
    #         "Content-Type": "application/json"
    #     }

    #     # Note: GPT-4V can analyze images but typically returns text descriptions
    #     # For image generation, you might need DALL-E or similar
    #     # This is a placeholder for vision analysis
    #     data = {
    #         "model": "gpt-4.1-mini",
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": prompt},
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {
    #                             "url": f"data:image/png;base64,{base64_image}"
    #                         }
    #                     }
    #                 ]
    #             }
    #         ],
    #         "max_tokens": 5000
    #     }

    #     response = requests.post(
    #         "https://api.openai.com/v1/images/generations",
    #         headers=headers,
    #         json=data
    #     )
    #     response.raise_for_status()

    #     result = response.json()
    #     description = result["choices"][0]["message"]["content"]
    #     print(f"VLM Analysis: {result}")

    #     print(f"VLM Analysis: {description}") 
    #     return self._create_analysis_image(description, output_filename)
        
    #     # # Check if this is a table transposition task
    #     # if "transpose" in prompt.lower() and "table" in prompt.lower():
    #     #     return self._generate_transposed_table_image_openai(description, output_filename)
    #     # else:
    #     #     # For other analysis tasks, create an image with the analysis results
    #     #     return self._create_analysis_image(description, output_filename)

    def _process_claude_vlm(self, image_path: str, prompt: str, output_filename: str) -> str:
        """
        Process image using Claude Vision.

        Args:
            image_path (str): Path to input image
            prompt (str): Text prompt
            output_filename (str): Output filename

        Returns:
            str: Path to output image
        """
        base64_image = self.encode_image_to_base64(image_path)

        headers = {
            "x-api-key": self.claude_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        data = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1000,
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
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        response.raise_for_status()

        result = response.json()
        description = result["content"][0]["text"]

        print(f"VLM Analysis: {description}")
        return self._create_analysis_image(description, output_filename)

        # # For table transposition, generate transposed image
        # if "transpose" in prompt.lower() and "table" in prompt.lower():
        #     return self._create_transposed_table_placeholder(description, output_filename)
        # else:
        #     # Create analysis image for other tasks
        #     return self._create_analysis_image(description, output_filename)

    def _generate_transposed_table_image_openai(self, analysis: str, output_filename: str) -> str:
        """
        Generate a transposed table image using DALL-E based on VLM analysis.

        Args:
            analysis (str): VLM analysis of the original table
            output_filename (str): Output filename

        Returns:
            str: Path to generated transposed image
        """
        try:
            # Create a prompt for DALL-E to generate the transposed table
            dalle_prompt = f"""
            Create a clear, professional table image showing the transposed version of this table.
            The table should have proper borders, readable text, and professional formatting.
            Make it look like a spreadsheet or database table with clear rows and columns.

            Analysis of original table: {analysis}

            Generate an image of the transposed table (rows become columns, columns become rows).
            Use a clean, modern design with clear cell borders and readable text.
            The table should be well-structured and easy to read.
            """

            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "dall-e-3",
                "prompt": dalle_prompt,
                "size": "1024x1024",
                "quality": "standard",
                "n": 1,
                "response_format": "b64_json"
            }

            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=data
            )
            response.raise_for_status()

            result = response.json()
            base64_image = result["data"][0]["b64_json"]

            # Decode the base64 image
            image = self.decode_image_from_base64(base64_image)

            # Save the image
            output_path = os.path.join(self.output_dir, output_filename)
            image.save(output_path)

            print(f"✓ Generated transposed table image with DALL-E: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Failed to generate transposed image with DALL-E: {e}")
            # Fallback: create a placeholder image
            return self._create_transposed_table_placeholder(analysis, output_filename)

    def _create_transposed_table_placeholder(self, analysis: str, output_filename: str) -> str:
        """
        Create a placeholder transposed table image when DALL-E is not available.

        Args:
            analysis (str): VLM analysis text
            output_filename (str): Output filename

        Returns:
            str: Path to placeholder image
        """
        try:
            # Create a simple image with transposed table visualization
            img = Image.new('RGB', (600, 400), color='white')
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()

            # Add title
            draw.text((50, 20), "TRANSPOSED TABLE RESULT", fill='black', font=font)

            # Add table structure visualization
            draw.rectangle([50, 60, 550, 350], outline='black', width=2)

            # Draw grid lines for transposed table
            for i in range(1, 4):
                y = 60 + i * 60
                draw.line([50, y, 550, y], fill='black', width=1)

            for i in range(1, 3):
                x = 50 + i * 250
                draw.line([x, 60, x, 350], fill='black', width=1)

            # Add sample transposed data labels
            draw.text((70, 80), "Original Columns", fill='black', font=font)
            draw.text((320, 80), "Become Rows", fill='black', font=font)
            draw.text((70, 140), "Original Rows", fill='black', font=font)
            draw.text((320, 140), "Become Columns", fill='black', font=font)

            # Add analysis summary
            analysis_summary = analysis[:100] + "..." if len(analysis) > 100 else analysis
            draw.text((70, 220), f"VLM Analysis: {analysis_summary}", fill='black', font=font)

            output_path = os.path.join(self.output_dir, output_filename)
            img.save(output_path)
            print(f"✓ Created transposed table placeholder image: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Failed to create placeholder image: {e}")
            # Final fallback
            img = Image.new('RGB', (400, 300), color='lightblue')
            output_path = os.path.join(self.output_dir, output_filename)
            img.save(output_path)
            return output_path

    def _create_analysis_image(self, analysis: str, output_filename: str) -> str:
        """
        Create an image containing the VLM analysis results.

        Args:
            analysis (str): VLM analysis text
            output_filename (str): Output filename

        Returns:
            str: Path to analysis image
        """
        try:
            # Create an image to display the analysis
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            # Add title
            draw.text((50, 30), "VLM IMAGE ANALYSIS RESULTS", fill='black', font=font)

            # Add analysis text with word wrapping
            y = 80
            words = analysis.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] < 700:  # Line width limit
                    line = test_line
                else:
                    draw.text((50, y), line, fill='black', font=font)
                    y += 30
                    line = word + " "
                    if y > 550:  # Page height limit
                        draw.text((50, y), "...", fill='black', font=font)
                        break
            if line and y <= 550:
                draw.text((50, y), line, fill='black', font=font)

            output_path = os.path.join(self.output_dir, output_filename)
            img.save(output_path)
            print(f"✓ Created analysis results image: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Failed to create analysis image: {e}")
            # Fallback
            img = Image.new('RGB', (400, 300), color='lightgray')
            output_path = os.path.join(self.output_dir, output_filename)
            img.save(output_path)
            return output_path

    def _process_openrouter_vlm(self, image_path: str, prompt: str, output_filename: str) -> str:
        """
        Process image using OpenRouter VLM provider.

        Args:
            image_path (str): Path to input image
            prompt (str): Text prompt
            output_filename (str): Output filename

        Returns:
            str: Path to output image
        """
        base64_image = self.encode_image_to_base64(image_path)

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", ""),
            "X-Title": os.getenv("OPENROUTER_TITLE", "")
        }

        # Use a vision-capable model through OpenRouter
        data = {
            "model": "openai/gpt-4-vision-preview",  # or other vision models available
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }

        response = requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()

        result = response.json()
        description = result["choices"][0]["message"]["content"]

        print(f"OpenRouter VLM Analysis: {description}")

        # Create analysis image
        return self._create_analysis_image(description, output_filename)

    def _process_huggingface_vlm(self, image_path: str, prompt: str, output_filename: str) -> str:
        """
        Process image using HuggingFace VLM provider.

        Args:
            image_path (str): Path to input image
            prompt (str): Text prompt
            output_filename (str): Output filename

        Returns:
            str: Path to output image
        """
        base64_image = self.encode_image_to_base64(image_path)

        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }

        # Use a vision-capable model (e.g., LLaVA, BLIP, or similar)
        # Note: This is a generic implementation - specific model requirements may vary
        model_id = "Salesforce/blip-image-captioning-base"  # Example model, adjust as needed

        data = {
            "inputs": {
                "image": base64_image,
                "text": prompt
            },
            "parameters": {
                "max_length": 100,
                "temperature": 0.7
            }
        }

        response = requests.post(
            f"{self.hf_base_url}/{model_id}",
            headers=headers,
            json=data
        )
        response.raise_for_status()

        result = response.json()

        # Handle different response formats from different models
        if isinstance(result, list) and result:
            description = result[0].get("generated_text", str(result[0]))
        else:
            description = str(result)

        print(f"HuggingFace VLM Analysis: {description}")

        # Create analysis image
        return self._create_analysis_image(description, output_filename)

    def _process_ollama_vlm(self, image_path: str, prompt: str, output_filename: str) -> str:
        """
        Process image using Ollama local VLM provider.

        Args:
            image_path (str): Path to input image
            prompt (str): Text prompt
            output_filename (str): Output filename

        Returns:
            str: Path to output image
        """
        base64_image = self.encode_image_to_base64(image_path)

        # Use vision-capable models like llava, bakllava, etc.
        data = {
            "model": self.model if hasattr(self, 'model') else "llava",  # Default to llava if no model specified
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "temperature": 0.7
        }

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=data,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            description = result.get("response", "")

            print(f"Ollama VLM Analysis: {description}")

        except requests.exceptions.ConnectionError:
            print(f"Could not connect to Ollama at {self.ollama_base_url}")
            print("Make sure Ollama is running: ollama serve")
            description = "Ollama connection failed"
        except Exception as e:
            print(f"Ollama VLM processing error: {e}")
            description = f"Error: {e}"

        # Create analysis image
        return self._create_analysis_image(description, output_filename)

    def transpose_table_image(self, image_path: str, output_filename: str = "transposed_table.png") -> str:
        """
        Transpose a table image using VLM processing.

        Args:
            image_path (str): Path to table image
            output_filename (str): Output filename for transposed table

        Returns:
            str: Path to transposed table image
        """
        with Image.open(image_path) as im:
            in_w, in_h = im.size
            
            target_w, target_h = in_h, in_w  # transpose
            target_ratio = target_w / target_h if target_h else 1.0  # width/height
        
        try:
            # Load the table transpose prompt template
            prompt = self.render_template("table_transpose.txt", target_width=target_w, target_height=target_h, target_ratio=target_ratio)

            # Process with VLM
            return self.process_with_vlm(image_path, prompt, output_filename)

        except Exception as e:
            print(f"Table transposition failed: {e}")
            # Fallback: return original image
            return self.save_image(self.load_image(image_path), output_filename)


    def process_with_template(self, image_path: str, template_name: str, output_filename: str = "output.png", **kwargs) -> str:
        """
        Process image using a template-based prompt.

        Args:
            image_path (str): Path to input image
            template_name (str): Template file name
            output_filename (str): Output filename
            **kwargs: Variables for template rendering

        Returns:
            str: Path to output edited image
        """
        prompt = self.render_template(template_name, **kwargs)
        return self.process_with_prompt(image_path, prompt, output_filename)

    def list_available_templates(self) -> list:
        """
        List all available image prompt templates.

        Returns:
            list: List of template file names
        """
        template_dir = Path(__file__).parent.parent / "templates" / "image_prompts"
        if template_dir.exists():
            return [f.name for f in template_dir.glob("*.txt")]
        return []


if __name__ == "__main__":
    # Example usage
    editor = ImagePromptEditor()
    print("Available templates:", editor.list_available_templates())
    print("Image Editor initialized. Use ImagePromptEditor class to process images.")
