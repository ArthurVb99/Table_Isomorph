"""
Example: Multi-provider LLM with Data Transformation
Demonstrates reading files, analyzing data, and applying permutations.
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from track_1_llm_prompt.llm_processor import LLMPromptProcessor
from track_1_llm_prompt.file_reader import DataFileReader
from track_1_llm_prompt.data_transformer import DataTransformer
from track_2_image_editing.image_editor import ImagePromptEditor


def example_openai_with_permutation():
    """Example: Use OpenAI to analyze table and apply permutations."""
    print("=" * 60)
    print("EXAMPLE 1: OpenAI Provider with Table Permutation")
    print("=" * 60)

    # Sample data
    data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "NYC"},
        {"id": 2, "name": "Bob", "age": 25, "city": "LA"},
        {"id": 3, "name": "Charlie", "age": 35, "city": "Chicago"},
    ]

    # Initialize processor with OpenAI
    try:
        processor = LLMPromptProcessor(provider="openai", model="gpt-3.5-turbo")
        print(f"\n✓ Connected to OpenAI")

        # Transform data
        transformer = DataTransformer(data)
        print(f"✓ Original data loaded: {len(transformer.df)} rows, {len(transformer.df.columns)} columns")

        # Apply permutations
        transformer.swap_rows_by_position(1, 2)
        transformer.swap_columns("name", "city")
        print(f"✓ Applied permutations: row 1↔2, column name↔city")

        # Generate LLM prompt for analysis
        response = processor.process_template(
            "table_permutation.txt",
            data=json.dumps(transformer.to_dict()),
            operation_type="both",
            num_permutations=2,
            constraints="Maintain data integrity"
        )
        print(f"\nLLM Response:\n{response[:300]}...")

    except Exception as e:
        print(f"✗ OpenAI Error: {e}")


def example_ollama_local():
    """Example: Use local Ollama server."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Local Ollama Provider")
    print("=" * 60)

    try:
        processor = LLMPromptProcessor(
            provider="ollama",
            model="mistral",  # or "neural-chat", "llama2", etc.
            base_url="http://localhost:11434"
        )
        print(f"✓ Connected to Ollama at http://localhost:11434")

        # Simple prompt
        response = processor.process_prompt("What is data transformation?")
        print(f"\nOllama Response:\n{response[:300]}...")

    except Exception as e:
        print(f"✗ Ollama Error: {e}")
        print("  → Make sure Ollama is running: ollama serve")


def example_huggingface():
    """Example: Use HuggingFace API."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: HuggingFace Provider")
    print("=" * 60)

    try:
        processor = LLMPromptProcessor(
            provider="huggingface",
            model="meta-llama/Llama-2-7b-chat-hf"
        )
        print(f"✓ Connected to HuggingFace")

        response = processor.process_template(
            "generic_question.txt",
            question="How to transform table structures?"
        )
        print(f"\nHuggingFace Response:\n{response[:300]}...")

    except Exception as e:
        print(f"✗ HuggingFace Error: {e}")


def example_openrouter():
    """Example: Use OpenRouter (cloud aggregator)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: OpenRouter Provider")
    print("=" * 60)

    try:
        processor = LLMPromptProcessor(
            provider="openrouter",
            model="openai/gpt-3.5-turbo"
        )
        print(f"✓ Connected to OpenRouter")

        response = processor.process_prompt("Explain table permutations in 2 sentences.")
        print(f"\nOpenRouter Response:\n{response}")

    except Exception as e:
        print(f"✗ OpenRouter Error: {e}")


def example_file_reading_and_transformation():
    """Example: Read file, analyze with LLM, then transform."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: File Reading & Data Transformation")
    print("=" * 60)

    # Create sample JSON file
    sample_data = [
        {"id": 1, "name": "Alice", "score": 95},
        {"id": 2, "name": "Bob", "score": 87},
        {"id": 3, "name": "Charlie", "score": 92},
    ]

    sample_file = "sample_data.json"
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f)

    try:
        # Read file
        data = DataFileReader.read_file(sample_file)
        print(f"✓ Read file: {sample_file}")
        print(f"  Data: {data}")

        # Transform
        transformer = DataTransformer(data)
        transformer.swap_rows_by_position(1, 2)
        transformer.swap_columns("name", "score")
        print(f"✓ Transformed: rows 1↔2, columns name↔score")
        print(f"  Result: {transformer.to_dict()}")

    except Exception as e:
        print(f"✗ Error: {e}")


def example_batch_analysis():
    """Example: Batch analysis of multiple items."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Batch Permutation Analysis")
    print("=" * 60)

    prompts = [
        "What is row-column permutation useful for?",
        "How does table transformation work?",
        "Give an example of data permutation",
    ]

    try:
        processor = LLMPromptProcessor(provider="openai", model="gpt-3.5-turbo")
        print(f"✓ Processing {len(prompts)} prompts...")

        responses = processor.process_batch_prompts(prompts)
        for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
            print(f"\n[{i}] {prompt}")
            print(f"    → {response[:100]}...")

    except Exception as e:
        print(f"✗ Error: {e}")


def example_advanced_permutations():
    """Example: Advanced permutation operations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Advanced Permutation Operations")
    print("=" * 60)

    data = [
        {"id": 1, "name": "Alice", "age": 30, "dept": "HR", "salary": 50000},
        {"id": 2, "name": "Bob", "age": 25, "dept": "IT", "salary": 60000},
        {"id": 3, "name": "Charlie", "age": 35, "dept": "Sales", "salary": 55000},
        {"id": 4, "name": "Diana", "age": 28, "dept": "HR", "salary": 52000},
    ]

    transformer = DataTransformer(data)
    print(f"✓ Original data: {len(transformer.df)} rows")
    transformer.display()

    # Complex permutations
    permutations = [
        {'type': 'row', 'first': 1, 'second': 3},
        {'type': 'column', 'first': 'name', 'second': 'dept'},
        {'type': 'row', 'first': 2, 'second': 4},
    ]

    transformer.apply_row_column_permutations(permutations)
    print(f"\n✓ After applying {len(permutations)} permutations:")
    transformer.display()

    # Export results
    result_json = transformer.to_json()
    print(f"\n✓ Exported as JSON:")
    print(result_json)


def example_visual_table_permutation():
    """Example: Visual table permutation using VLM analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: VISUAL TABLE PERMUTATION WITH VLM")
    print("=" * 60)

    # Create sample table data
    table_data = [
        {"Employee": "Alice", "Department": "HR", "Salary": 50000, "Experience": 5},
        {"Employee": "Bob", "Department": "IT", "Salary": 60000, "Experience": 3},
        {"Employee": "Charlie", "Department": "Sales", "Salary": 55000, "Experience": 7},
        {"Employee": "Diana", "Department": "HR", "Salary": 52000, "Experience": 4},
    ]

    print("Sample table data:")
    for row in table_data:
        print(f"  {row}")

    # Step 1: LLM Analysis for permutation strategy
    print("\n1. LLM Analysis - Determining permutation strategy")
    try:
        llm_processor = LLMPromptProcessor(provider="openai", model="gpt-3.5-turbo")
        print("✓ LLM Processor initialized")

        # Get permutation suggestions from LLM
        permutation_prompt = f"""
        Analyze this table data and suggest meaningful permutations:

        Table Data:
        {json.dumps(table_data, indent=2)}

        Suggest 2-3 permutation operations that would:
        1. Group similar data together
        2. Sort by a logical criteria
        3. Rearrange for better analysis

        Provide specific row and column swaps with reasoning.
        """

        llm_response = llm_processor.process_prompt(permutation_prompt)
        print(f"LLM Strategy: {llm_response[:300]}...")

    except Exception as e:
        print(f"✗ LLM Analysis failed: {e}")
        llm_response = "Fallback: Swap rows 1-2 and columns Department-Salary"

    # Step 2: Apply permutations to data
    print("\n2. Applying Data Permutations")
    try:
        transformer = DataTransformer(table_data)

        # Apply suggested permutations
        transformer.swap_rows_by_position(0, 2)  # Alice ↔ Charlie
        transformer.swap_columns("Department", "Experience")  # Department ↔ Experience

        permuted_data = transformer.to_dict()
        print("✓ Applied permutations: rows 0↔2, columns Department↔Experience")
        print("Permuted data:")
        for row in permuted_data:
            print(f"  {row}")

    except Exception as e:
        print(f"✗ Data permutation failed: {e}")
        permuted_data = table_data

    # Step 3: Visual table processing with VLM (if image available)
    print("\n3. Visual Table Analysis with VLM")
    print("Note: This step requires actual table images for full VLM processing")

    try:
        # Initialize image editor with VLM support
        image_editor = ImagePromptEditor(vlm_provider="openai")
        print("✓ Image Editor with VLM initialized")

        # Show available image templates
        template_dir = Path("templates/image_prompts")
        if template_dir.exists():
            templates = list(template_dir.glob("*.txt"))
            print(f"Available visual templates: {[t.name for t in templates]}")

            # Load table transpose template
            if any("table_transpose" in t.name for t in templates):
                visual_prompt = image_editor.process_with_template("table_transpose.txt", {
                    "table_data": json.dumps(permuted_data),
                    "permutation_description": "Rows swapped: Alice↔Charlie, Columns: Department↔Experience"
                })
                print(f"Visual prompt generated: {visual_prompt[:200]}...")

                print("\nTo process actual table images, you would:")
                print("  1. Have a PNG/JPG image of the table")
                print("  2. Call: image_editor.transpose_table_image('table.png')")
                print("  3. VLM would analyze and generate transposed visual")

    except Exception as e:
        print(f"✗ Visual processing setup failed: {e}")

    # Step 4: Combined analysis
    print("\n4. Combined Text + Visual Analysis Results")
    try:
        # Use LLM to analyze the permutation results
        analysis_prompt = f"""
        Analyze the permutation results:

        Original data:
        {json.dumps(table_data, indent=2)}

        After permutation:
        {json.dumps(permuted_data, indent=2)}

        Explain what changed and why this permutation might be useful.
        """

        final_analysis = llm_processor.process_prompt(analysis_prompt)
        print(f"Combined Analysis: {final_analysis[:400]}...")

    except Exception as e:
        print(f"✗ Combined analysis failed: {e}")


def example_template_based_image_processing():
    """Example: Template-based image processing with dynamic prompts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: TEMPLATE-BASED IMAGE PROCESSING")
    print("=" * 60)

    try:
        # Initialize image editor
        image_editor = ImagePromptEditor(vlm_provider="openai")
        print("✓ Image Editor initialized")

        # List available templates
        templates = image_editor.list_available_templates()
        print(f"Available templates: {templates}")

        # Example 1: Basic edits template
        print("\n1. Basic Edits Template")
        try:
            # Create a simple test image (since we don't have real images)
            test_image = Image.new('RGB', (400, 300), color='lightblue')
            # Add some text to simulate a table
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(test_image)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            draw.text((50, 50), "Sample Table", fill='black', font=font)
            draw.text((50, 100), "Name | Age | City", fill='black', font=font)
            draw.text((50, 130), "Alice | 30 | NYC", fill='black', font=font)
            draw.text((50, 160), "Bob | 25 | LA", fill='black', font=font)

            # Save test image
            test_image_path = "test_table.png"
            test_image.save(test_image_path)
            print(f"✓ Created test image: {test_image_path}")

            # Use basic_edits template
            operations = "brighten the image, increase contrast, and apply slight blur"
            result_path = image_editor.process_with_template(
                test_image_path,
                "basic_edits.txt",
                "basic_edits_output.png",
                operations=operations
            )
            print(f"✓ Processed with basic_edits template: {result_path}")

        except Exception as e:
            print(f"✗ Basic edits example failed: {e}")

        # Example 2: Enhancement template
        print("\n2. Enhancement Template")
        try:
            enhancement_result = image_editor.process_with_template(
                test_image_path,
                "enhancement.txt",
                "enhanced_output.png",
                brightness="increase by 20%",
                contrast="high",
                saturation="vivid colors",
                apply="professional look"
            )
            print(f"✓ Processed with enhancement template: {enhancement_result}")

        except Exception as e:
            print(f"✗ Enhancement example failed: {e}")

        # Example 3: Table transpose template (text-based for now)
        print("\n3. Table Transpose Template")
        try:
            # Show how the template would be used for table transposition
            transpose_prompt = image_editor.render_template("table_transpose.txt", {
                "table_data": json.dumps([
                    {"Name": "Alice", "Age": 30, "City": "NYC"},
                    {"Name": "Bob", "Age": 25, "City": "LA"}
                ]),
                "permutation_description": "Swap rows and transpose columns"
            })
            print("Table transpose prompt generated:")
            print(transpose_prompt[:200] + "...")

            # For actual image processing (would need real table image):
            print("\nTo process actual table images:")
            print("  result = image_editor.process_with_template(")
            print("      'table_image.png',")
            print("      'table_transpose.txt',")
            print("      'transposed_table.png',")
            print("      table_data=json.dumps(table_data),")
            print("      permutation_description='specific changes'")
            print("  )")

        except Exception as e:
            print(f"✗ Table transpose example failed: {e}")

        # Example 4: Dynamic template rendering
        print("\n4. Dynamic Template Rendering")
        try:
            # Show how templates can be rendered without processing images
            custom_prompt = image_editor.render_template("enhancement.txt", {
                "brightness": "dramatic increase",
                "contrast": "maximum",
                "saturation": "neon colors",
                "apply": "retro 80s style"
            })
            print("Custom rendered prompt:")
            print(custom_prompt)

        except Exception as e:
            print(f"✗ Dynamic rendering example failed: {e}")

        # Cleanup
        try:
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                print(f"\n✓ Cleaned up test image: {test_image_path}")
        except:
            pass

    except Exception as e:
        print(f"✗ Template-based processing example failed: {e}")


def example_process_with_vlm():
    """Example: Using process_with_vlm for Vision-Language Model image analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: PROCESS_WITH_VLM - VISION-LANGUAGE MODEL ANALYSIS")
    print("=" * 60)

    # Create a simple test image (simulating a table)
    print("1. Creating test table image")
    try:
        test_image = Image.new('RGB', (500, 300), color='white')
        draw = ImageDraw.Draw(test_image)

        # Draw a simple table structure
        draw.rectangle([50, 50, 450, 250], outline='black', width=2)

        # Draw table lines
        for i in range(1, 4):
            y = 50 + i * 50
            draw.line([50, y, 450, y], fill='black', width=1)

        for i in range(1, 4):
            x = 50 + i * 100
            draw.line([x, 50, x, 250], fill='black', width=1)

        # Add table headers and data
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        headers = ["Name", "Age", "City", "Salary"]
        data = [
            ["Alice", "30", "NYC", "75000"],
            ["Bob", "25", "LA", "65000"],
            ["Charlie", "35", "Chicago", "80000"]
        ]

        # Draw headers
        for i, header in enumerate(headers):
            x = 50 + i * 100 + 10
            draw.text((x, 60), header, fill='black', font=font)

        # Draw data
        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                x = 50 + col_idx * 100 + 10
                y = 110 + row_idx * 50
                draw.text((x, y), cell, fill='black', font=font)

        # Save test image
        test_image_path = "test_table_image.png"
        test_image.save(test_image_path)
        print(f"✓ Created test table image: {test_image_path}")

    except Exception as e:
        print(f"✗ Failed to create test image: {e}")
        return

    # Example prompts for different VLM tasks
    vlm_prompts = {
        "table_analysis": """
        Analyze this table image and provide:
        1. The table structure (number of rows and columns)
        2. Column headers
        3. Data types in each column
        4. Any patterns or insights you can observe
        """,

        "table_transposition": """
        This image contains a table. Describe how you would transpose it
        (swap rows and columns) and explain what the transposed table would look like.
        Provide the transposed data structure.
        """,

        "data_insights": """
        Look at this employee table and provide insights about:
        1. Age distribution
        2. Salary ranges
        3. City distribution
        4. Any correlations you can identify
        """
    }

    # Test different VLM providers
    providers_to_test = ["openai", "claude", "openrouter", "huggingface", "ollama"]

    for provider in providers_to_test:
        print(f"\n2. Testing {provider.upper()} VLM Provider")
        try:
            # Initialize image editor with specific VLM provider
            image_editor = ImagePromptEditor(vlm_provider=provider)
            print(f"✓ Initialized {provider} VLM provider")

            # Process with table analysis prompt
            result_path = image_editor.process_with_vlm(
                test_image_path,
                vlm_prompts["table_analysis"],
                f"vlm_output_{provider}.png"
            )
            print(f"✓ VLM analysis completed, output saved to: {result_path}")

        except ValueError as e:
            if "API_KEY" in str(e) or "not found" in str(e):
                print(f"⚠ {provider} skipped: API key not configured - {e}")
            else:
                print(f"✗ {provider} failed: {e}")
        except Exception as e:
            print(f"✗ {provider} error: {e}")

    # Example 3: Using different prompts
    print("\n3. Testing Different VLM Prompts")
    try:
        # Use OpenAI if available, otherwise skip
        image_editor = ImagePromptEditor(vlm_provider="openai")

        for prompt_name, prompt_text in vlm_prompts.items():
            print(f"\n--- {prompt_name.upper()} ---")
            try:
                result_path = image_editor.process_with_vlm(
                    test_image_path,
                    prompt_text,
                    f"vlm_{prompt_name}.png"
                )
                print(f"✓ Completed {prompt_name} analysis")
            except Exception as e:
                print(f"✗ {prompt_name} failed: {e}")

    except Exception as e:
        print(f"⚠ Skipping prompt examples: {e}")

    # Example 4: Integration with table transposition
    print("\n4. VLM-Assisted Table Transposition")
    try:
        image_editor = ImagePromptEditor(vlm_provider="openai")

        # Use the transpose_table_image method (which uses VLM internally)
        transposed_path = image_editor.transpose_table_image(
            test_image_path,
            "vlm_transposed_table.png"
        )
        print(f"✓ Table transposition completed: {transposed_path}")

    except Exception as e:
        print(f"⚠ Table transposition example skipped: {e}")

    # Cleanup
    try:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"\n✓ Cleaned up test image: {test_image_path}")
    except:
        pass

    print("\n" + "=" * 60)
    print("VLM PROCESSING EXAMPLES COMPLETED")
    print("=" * 60)
    print("Note: Actual VLM processing requires API keys in .env file:")
    print("  OPENAI_API_KEY=your_openai_key")
    print("  ANTHROPIC_API_KEY=your_claude_key")
    print("  OPENROUTER_API_KEY=your_openrouter_key")
    print("  HUGGINGFACE_API_KEY=your_hf_key")
    print("  OLLAMA_BASE_URL=http://localhost:11434 (for local Ollama)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TABLE TRANSFORMATION & MULTI-PROVIDER LLM EXAMPLES")
    print("=" * 60)

    # Run examples (comment out providers not configured)
    try:
        example_openai_with_permutation()
    except Exception as e:
        print(f"Skipping OpenAI example: {e}")

    try:
        example_file_reading_and_transformation()
    except Exception as e:
        print(f"Skipping file reading example: {e}")

    try:
        example_advanced_permutations()
    except Exception as e:
        print(f"Skipping advanced example: {e}")

    try:
        example_visual_table_permutation()
    except Exception as e:
        print(f"Skipping visual example: {e}")

    try:
        example_template_based_image_processing()
    except Exception as e:
        print(f"Skipping template example: {e}")

    try:
        example_process_with_vlm()
    except Exception as e:
        print(f"Skipping VLM example: {e}")

    # Uncomment to test other providers:
    # example_ollama_local()
    # example_huggingface()
    # example_openrouter()
    # example_batch_analysis()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
