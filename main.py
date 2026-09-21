#!/usr/bin/env python3
"""
Table Structure Recognition Project - Main Entry Point
Combines LLM prompt processing and image-based table editing capabilities.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Import project modules
from track_1_llm_prompt.llm_processor import LLMPromptProcessor
from track_1_llm_prompt.file_reader import DataFileReader
from track_1_llm_prompt.data_transformer import DataTransformer
from track_2_image_editing.image_editor import ImagePromptEditor
from track_4_image_edit_and_project.image_editor_projection import ImagePromptEditorProjection

class TableStructureRecognitionApp:
    """Main application class for table structure recognition."""

    def __init__(self):
        """Initialize the application."""
        self.llm_processor: Optional[LLMPromptProcessor] = None
        self.image_editor: Optional[ImagePromptEditor] = None
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from environment or defaults."""
        config = {
            "llm_provider": os.getenv("DEFAULT_LLM_PROVIDER", "openai"),
            "llm_model": os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo"),
            "vlm_provider": os.getenv("DEFAULT_VLM_PROVIDER", "openai"),
            "output_dir": "/data/brussel/vo/000/bvo00018/vsc11306/cross-modal_experiments/results",
            "sample_data_file": "sample_data.json"
        }

        # Create output directory if it doesn't exist
        Path(config["output_dir"]).mkdir(exist_ok=True)

        return config

    def setup_llm_processor(self, provider: str = None, model: str = None) -> bool:
        """Setup LLM processor with specified provider."""
        try:
            provider = provider or self.config["llm_provider"]
            model = model or self.config["llm_model"]

            self.llm_processor = LLMPromptProcessor(provider=provider, model=model)
            print(f"✓ LLM Processor initialized: {provider}/{model}")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize LLM processor: {e}")
            return False

    def setup_image_editor(self, vlm_provider: str = None) -> bool:
        """Setup image editor with VLM support."""
        try:
            vlm_provider = vlm_provider or self.config["vlm_provider"]
            self.image_editor = ImagePromptEditor(vlm_provider=vlm_provider)
            print(f"✓ Image Editor initialized with VLM: {vlm_provider}")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize image editor: {e}")
            return False

    def create_sample_data(self) -> str:
        """Create sample table data for testing."""
        sample_data = [
            {"id": 1, "name": "Alice", "age": 30, "city": "New York", "salary": 75000},
            {"id": 2, "name": "Bob", "age": 25, "city": "Los Angeles", "salary": 65000},
            {"id": 3, "name": "Charlie", "age": 35, "city": "Chicago", "salary": 80000},
            {"id": 4, "name": "Diana", "age": 28, "city": "Houston", "salary": 70000},
        ]

        filename = self.config["sample_data_file"]
        with open(filename, 'w') as f:
            json.dump(sample_data, f, indent=2)

        print(f"✓ Created sample data: {filename}")
        return filename

    def run_llm_examples(self):
        """Run LLM-based table processing examples."""
        print("\n" + "="*60)
        print("TRACK 1: LLM-BASED TABLE PROCESSING")
        print("="*60)

        if not self.setup_llm_processor():
            print("Skipping LLM examples due to initialization failure.")
            return

        # Create sample data
        data_file = self.create_sample_data()

        # Example 1: Basic table analysis
        print("\n1. Basic Table Analysis")
        try:
            data = DataFileReader.read_file(data_file)
            transformer = DataTransformer(data)

            response = self.llm_processor.process_template(
                "data_analysis.txt",
                table_data=json.dumps(data),
                analysis_type="summary"
            )
            print(f"Analysis: {response[:200]}...")
        except Exception as e:
            print(f"✗ Analysis failed: {e}")

        # Example 2: Table transformation
        print("\n2. Table Transformation")
        try:
            transformer = DataTransformer(data)
            transformer.swap_rows_by_position(0, 2)
            transformer.swap_columns("name", "city")

            response = self.llm_processor.process_template(
                "table_permutation.txt",
                data=json.dumps(transformer.to_dict()),
                operation_type="both",
                num_permutations=2
            )
            print(f"Transformation result: {response[:200]}...")
        except Exception as e:
            print(f"✗ Transformation failed: {e}")

        # Example 3: Code generation
        print("\n3. Code Generation for Data Processing")
        try:
            response = self.llm_processor.process_template(
                "code_help.txt",
                task="Generate Python code to transpose a pandas DataFrame",
                language="python"
            )
            print(f"Generated code: {response[:300]}...")
        except Exception as e:
            print(f"✗ Code generation failed: {e}")

    def run_image_examples(self):
        """Run image-based table processing examples."""
        print("\n" + "="*60)
        print("TRACK 2: IMAGE-BASED TABLE PROCESSING")
        print("="*60)

        if not self.setup_image_editor():
            print("Skipping image examples due to initialization failure.")
            return

        # Note: These examples require actual image files
        # For demonstration, we'll show the API structure

        print("\n1. Basic Image Editing")
        print("   (Requires image file - example structure shown)")

        print("\n2. Table Transposition via VLM")
        print("   (Requires table image - VLM analysis capability)")

        print("\n3. Template-based Image Processing")
        try:
            # Show available templates
            template_dir = Path("templates/image_prompts")
            if template_dir.exists():
                templates = list(template_dir.glob("*.txt"))
                print(f"   Available templates: {[t.name for t in templates]}")

                # Example template processing (without actual image)
                if templates:
                    template_name = templates[0].name
                    prompt = self.image_editor.process_with_template(template_name, {})
                    print(f"   Sample prompt from {template_name}: {prompt[:100]}...")
        except Exception as e:
            print(f"✗ Template processing failed: {e}")

    def run_interactive_mode(self):
        """Run interactive mode for custom processing."""
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)

        while True:
            print("\nChoose an option:")
            print("1. Process text data with LLM")
            print("2. Process image with VLM")
            print("3. Transform table data")
            print("4. Generate code")
            print("5. Back to main menu")

            choice = input("\nEnter choice (1-5): ").strip()

            if choice == "1":
                self.interactive_llm_processing()
            elif choice == "2":
                self.interactive_image_processing()
            elif choice == "3":
                self.interactive_data_transformation()
            elif choice == "4":
                self.interactive_code_generation()
            elif choice == "5":
                break
            else:
                print("Invalid choice. Please try again.")

    def interactive_llm_processing(self):
        """Interactive LLM text processing."""
        if not self.setup_llm_processor():
            return

        prompt = input("Enter your prompt: ").strip()
        if prompt:
            try:
                response = self.llm_processor.process_prompt(prompt)
                print(f"\nResponse:\n{response}")
            except Exception as e:
                print(f"✗ Processing failed: {e}")

    def interactive_image_processing(self):
        """Interactive image processing (placeholder)."""
        print("Image processing requires image files.")
        print("Please ensure you have table images in the workspace.")
        # This would be expanded with actual image processing logic

    def interactive_data_transformation(self):
        """Interactive data transformation."""
        data_file = input("Enter data file path (or press Enter for sample): ").strip()
        if not data_file:
            data_file = self.create_sample_data()

        try:
            data = DataFileReader.read_file(data_file)
            transformer = DataTransformer(data)

            print(f"Loaded data with {len(transformer.df)} rows, {len(transformer.df.columns)} columns")

            # Simple transformation options
            print("Available transformations:")
            print("1. Swap rows")
            print("2. Swap columns")
            print("3. Display current data")

            choice = input("Choose transformation (1-3): ").strip()

            if choice == "1":
                row1 = int(input("First row index: "))
                row2 = int(input("Second row index: "))
                transformer.swap_rows_by_position(row1, row2)
                print("Rows swapped successfully")
            elif choice == "2":
                col1 = input("First column name: ")
                col2 = input("Second column name: ")
                transformer.swap_columns(col1, col2)
                print("Columns swapped successfully")
            elif choice == "3":
                transformer.display()

        except Exception as e:
            print(f"✗ Transformation failed: {e}")

    def interactive_code_generation(self):
        """Interactive code generation."""
        if not self.setup_llm_processor():
            return

        task = input("Describe the code you need: ").strip()
        language = input("Programming language (default: python): ").strip() or "python"

        if task:
            try:
                response = self.llm_processor.process_template(
                    "code_help.txt",
                    task=task,
                    language=language
                )
                print(f"\nGenerated code:\n{response}")
            except Exception as e:
                print(f"✗ Code generation failed: {e}")

    def show_menu(self):
        """Display main menu."""
        print("\n" + "="*60)
        print("TABLE STRUCTURE RECOGNITION PROJECT")
        print("="*60)
        print("1. Run LLM Examples (Track 1)")
        print("2. Run Image Processing Examples (Track 2)")
        print("3. Interactive Mode")
        print("4. Setup & Configuration")
        print("5. Exit")
        print("="*60)

    def run_setup(self):
        """Run setup and configuration."""
        print("\n" + "="*60)
        print("SETUP & CONFIGURATION")
        print("="*60)

        print("Current configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        print("\nEnvironment variables to set:")
        print("  OPENAI_API_KEY - for OpenAI GPT models")
        print("  ANTHROPIC_API_KEY - for Claude models")
        print("  HUGGINGFACE_API_KEY - for HuggingFace models")
        print("  OPENROUTER_API_KEY - for OpenRouter access")

        print("\nMake sure requirements are installed:")
        print("  pip install -r requirements.txt")

    def main(self):
        """Main application loop."""
        print("Welcome to Table Structure Recognition Project!")

        while True:
            self.show_menu()
            choice = input("Enter your choice (1-5): ").strip()

            if choice == "1":
                self.run_llm_examples()
            elif choice == "2":
                self.run_image_examples()
            elif choice == "3":
                self.run_interactive_mode()
            elif choice == "4":
                self.run_setup()
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

            input("\nPress Enter to continue...")


def main():
    """Entry point for the application."""
    try:
        app = TableStructureRecognitionApp()
        app.main()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()