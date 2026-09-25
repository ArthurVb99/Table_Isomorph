"""
LLM Prompt Processor for Track 1
Handles prompt generation, processing, and LLM interactions with multiple providers.
Supports: OpenAI, Ollama (local), HuggingFace, OpenRouter
"""

import os
import json
import tiktoken
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
from typing import Any, Optional, List
# from langchain_community.llms import OpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage
import requests

from track_1_llm_prompt.validators import LLMTableModel

load_dotenv()


class LLMPromptProcessor:
    """
    Process and interact with Large Language Models using text prompts and templates.
    Supports multiple providers: OpenAI, Ollama, HuggingFace, OpenRouter
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        temperature: float =1,
        **kwargs
    ):
        """
        Initialize the LLM processor with specified provider.

        Args:
            provider (str): LLM provider - 'openai', 'ollama', 'huggingface', 'openrouter'
            model (str): Model name for the provider
            temperature (float): Creativity level (0-1)
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

        # Initialize LLM based on provider
        self._initialize_llm()

        # Setup Jinja2 template loader
        template_dir = Path(__file__).parent.parent / "templates" / "llm_prompts"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

    def _initialize_llm(self):
        """Initialize LLM client based on provider."""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "huggingface":
            self._init_huggingface()
        elif self.provider == "openrouter":
            self._init_openrouter()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _init_openai(self):
        """Initialize OpenAI provider."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.llm = ChatOpenAI(
            model_name=self.model,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key
        )

    def _init_ollama(self):
        """Initialize Ollama local provider."""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = None  # Will use requests directly

    def _init_huggingface(self):
        """Initialize HuggingFace provider."""
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

        self.hf_api_key = api_key
        self.hf_base_url = os.getenv("HUGGINGFACE_BASE_URL", "https://api-inference.huggingface.co/models")
        self.llm = None  # Will use requests directly

    def _init_openrouter(self):
        """Initialize OpenRouter provider."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.openrouter_api_key = api_key
        # Prefer the canonical API host and path used by OpenRouter services.
        # Allow override via OPENROUTER_BASE_URL environment variable.
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://api.openrouter.ai/v1")
        self.llm = None  # Will use requests directly

    def process_prompt(self, prompt: str, max_tokens: int = 50000) -> str:
        """
        Send a prompt to the LLM and get a response.

        Args:
            prompt (str): The text prompt to send
            max_tokens (int): Maximum tokens in response

        Returns:
            str: The LLM response text
        """
        try:
            if self.provider == "openai":
                # structured_llm = self.llm.with_structured_output(LLMTableModel)
                # response = structured_llm.invoke([HumanMessage(content=prompt)])
                # convert to JSON string
                # return response.json()
                # response = self.llm.invoke([HumanMessage(content=prompt)])
                # return response.content
                return self._process_openai_vlm(prompt)

            elif self.provider == "ollama":
                return self._process_ollama(prompt, max_tokens)

            elif self.provider == "huggingface":
                return self._process_huggingface(prompt, max_tokens)

            elif self.provider == "openrouter":
                return self._process_openrouter(prompt, max_tokens)

        except Exception as e:
            raise RuntimeError(f"Error processing prompt with {self.provider}: {str(e)}")
        
    def enforce_no_additional_properties(self, schema: Any) -> Any:
        """
        OpenAI Structured Outputs strict mode requires:
        - every object schema must set additionalProperties: false
        This recursively applies it.
        """
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                schema["additionalProperties"] = False
            for k, v in list(schema.items()):
                if isinstance(v, (dict, list)):
                    schema[k] = self.enforce_no_additional_properties(v)
            return schema
        if isinstance(schema, list):
            return [self.enforce_no_additional_properties(x) for x in schema]
        return schema

    def _process_openai_vlm(self, prompt: str) -> Optional[str]:
        """Process with OpenAI Vision."""

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        schema = LLMTableModel.model_json_schema()  # Pydantic v2
        schema = self.enforce_no_additional_properties(schema)
        payload = {
            "model": self.model, 
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
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
    
    def _process_ollama(self, prompt: str, max_tokens: int) -> str:
        """Process prompt using Ollama local server."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                },
                #timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.ollama_base_url}. "
                "Make sure Ollama is running: ollama serve"
            )

    def _process_huggingface(self, prompt: str, max_tokens: int) -> str:
        """Process prompt using HuggingFace API."""
        url = f"{self.hf_base_url}/{self.model}"
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_tokens,
                "temperature": self.temperature,
            }
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and result:
            return result[0].get("generated_text", "")
        return str(result)

    def _process_openrouter(self, prompt: str, max_tokens: int) -> str:
        """Process prompt using OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self.openrouter_base_url}/chat/completions"
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Provide clearer diagnostic info for 404/other errors.
            status = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            body = e.response.text if hasattr(e, 'response') and e.response is not None else ''
            raise RuntimeError(
                f"OpenRouter request failed ({status}) for URL {url}: {body}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter request error: {str(e)}") from e

        # Parse response safely
        try:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected OpenRouter response format: {e}. Raw: {response.text}") from e

    def load_template(self, template_name: str) -> str:
        """
        Load a prompt template from file.

        Args:
            template_name (str): Name of template file (e.g., 'generic_question.txt')

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

    def count_input_tokens(self, input_data, model: str = None) -> int:
        """
        Count tokens for input data (string or JSON).

        Args:
            input_data: String or JSON-serializable data.
            model (str): Model name (defaults to self.model).

        Returns:
            int: Token count.
        """
        if model is None:
            model = self.model

        # Convert JSON to string if needed
        if isinstance(input_data, (dict, list)):
            text = json.dumps(input_data)
        else:
            text = str(input_data)

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Use a general-purpose estimate for custom/local model names.
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)


    def process_template(self, template_name: str, max_tokens: int = 50000, **kwargs) -> str:
        """
        Process a prompt using a template.

        Args:
            template_name (str): Template file name
            max_tokens (int): Maximum tokens in response
            **kwargs: Variables for template rendering

        Returns:
            str: The LLM response text
        """
        prompt = self.render_template(template_name, **kwargs)
        return self.process_prompt(prompt, max_tokens)

    def process_batch_prompts(self, prompts: list, max_tokens: int = 500) -> list:
        """
        Process multiple prompts and return a list of responses.

        Args:
            prompts (list): List of text prompts
            max_tokens (int): Maximum tokens per response

        Returns:
            list: List of LLM responses
        """
        responses = []
        for prompt in prompts:
            response = self.process_prompt(prompt, max_tokens)
            responses.append(response)
        return responses

    def list_available_templates(self) -> list:
        """
        List all available prompt templates.

        Returns:
            list: List of template file names
        """
        template_dir = Path(__file__).parent.parent / "templates" / "llm_prompts"
        if template_dir.exists():
            return [f.name for f in template_dir.glob("*.txt")]
        return []

    @staticmethod
    def get_available_providers() -> List[str]:
        """
        Get list of available providers.

        Returns:
            list: Available provider names
        """
        return ["openai", "ollama", "huggingface", "openrouter"]


if __name__ == "__main__":
    # Example usage
    processor = LLMPromptProcessor(provider="openai", model="gpt-3.5-turbo")

    # List available templates
    print("Available templates:", processor.list_available_templates())

    # Use template with variables
    result = processor.process_template(
        "generic_question.txt",
        question="What is the capital of France?"
    )
    print(f"Response: {result}")
