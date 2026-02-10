# Measuring Cross-Modal Semantic Consistency between Symbolic and Visual Table Reasoning
This work implements a three-track AI system designed to measure the degree ($\epsilon$) of **cross-modal semantic consistency** in table reasoning across both visual and symbolic modalities. By using LangChain and template-based prompts with multi-provider LLM support, the work introduces **prompt-as-instruction isomorphism (PII)** as the main evalluation framework.

### Core Functionality and Methodology
The system operates through a closed-loop framework to measure cross-modal alignment:
- **Symbolic Track**: A Large Language Model (LLM) applies a structural transformation to a table in the symbolic domain (e.g., JSON, HTML or Token-Oriented Object). 
- **Visual Track**: A Vision-Language Model (VLM) performs a visually equivalent image-to-image edit on the source table image. 
- **TSR Integration**: The project uses Table Structure Recognition (TSR) to re-project edited images back into the symbolic domain. 
- **Evaluation**: Direct cross-modal structural comparisons are conducted using metrics such as TEDS and GriTS.

### Prerequisites
- Python 3.8 or higher
- pip or conda
- At least one LLM provider configured (see below)

### Create Virtual Environment

#### Using Conda (Windows, macOS, Linux)

```bash
# Create environment
conda create -n table_structure python=3.10

# Activate environment
conda activate table_structure

# Install dependencies
pip install -r requirements.txt
```

#### Using Python venv (Linux/macOS)

```bash
# Create environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Using Python venv (Windows PowerShell)

```bash
# Create environment
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root. You can copy the template:

```bash
cp .env.example .env
```

Then edit `.env` and add your configuration:

```
# ============================================================
# OPENAI PROVIDER
# ============================================================
OPENAI_API_KEY=sk-...your_openai_api_key_here...

# ============================================================
# OLLAMA PROVIDER (Local Server)
# ============================================================
OLLAMA_BASE_URL=http://localhost:11434

# ============================================================
# HUGGINGFACE PROVIDER
# ============================================================
HUGGINGFACE_API_KEY=hf_...your_huggingface_api_key_here...
HUGGINGFACE_BASE_URL=https://api-inference.huggingface.co/models

# ============================================================
# OPENROUTER PROVIDER
# ============================================================
OPENROUTER_API_KEY=sk-or-...your_openrouter_api_key_here...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# ============================================================
# CLAUDE PROVIDER
# ============================================================
ANTHROPIC_API_KEY=sk-ant-...your_anthropic_api_key_here...
```

**Important:** Only set the API keys for providers you plan to use. All base URLs have defaults.

---

## LLM Providers

### 1. **OpenAI** (Cloud - Recommended for beginners)

```bash
# Install
pip install -r requirements.txt

# Configure
# Set OPENAI_API_KEY in .env

# Usage
from track_1_llm_prompt.llm_processor import LLMPromptProcessor
processor = LLMPromptProcessor(provider="openai", model="gpt-3.5-turbo")
```

## Track 1: LLM Prompt Processor with Multi-Provider Support

**Directory:** `track_1_llm_prompt/`

Handles text-based prompt interactions with multiple LLM providers.

## Track 2: Image Prompt Editor with Templates

**Directory:** `track_2_image_editing/`

Processes images based on text prompts or templates and outputs edited images.

### Validation Integration

Track 3 uses the same validators as Track 1:

```python
from track_3_image_projection.image_projection import ImageProjectionProcessor
from track_1_llm_prompt.validators import validate_llm_output, compare_cell_count

processor = ImageProjectionProcessor()
model, error = processor.extract_table_structure("table.png")

if model:
    # Additional validation if needed
    validated_model, validation_error = validate_llm_output(model.model_dump())
    if validated_model:
        print("Structure validated successfully")
    else:
        print("Validation error:", validation_error)
```

### Batch Script

Use the provided batch script for large-scale processing:

```bash
# Set environment variables
set PATH_INPUT_IMAGES=path/to/image/directory
set PATH_OUTPUT_JSONL=extracted_structures.jsonl
set VLM_PROVIDER=openai
set MODEL=gpt5
set MAX_THREADS=4

# Run batch processing
python batch_structure_extraction.py
```

---

## Templates

### LLM Prompts (`templates/llm_prompts/`)

| Template | Purpose |
|---|---|
| table_permutation.txt | Analyze tables and generate permutations |
| sample_generation.txt | Generate sample data with transformations |

### Image Prompts (`templates/image_prompts/`)

| Template | Purpose |
|---|---|
| table_transpose.txt | Table transposition for image editing |
| table_structure_extraction.txt | Table structure recognition and JSON extraction |

---

## Evaluation: Measuring Structural Consistency
Once you have processed the dataset through the Symbolic ($L_f$) and Visual ($V_f$) tracks, use the provided evaluation script to calculate the TEDS (Table Edit Distance Similarity) and GriTS (Grid Table Similarity) scores.

1. Download the TableNetTab dataset and place it in the `data/` directory. Ensure the structure follows the required format:

```plaintext
data/
└── TableNetTab/
    ├── images/          # Source table images
    ├── ground_truth/    # Corresponding JSON/TOO files
    └── predictions/     # Outputs from VLM (Visual) and LLM (Symbolic)
```

2. Run the Evaluation Script

The `evaluate_metrics.py` script compares the ground truth with the model outputs to quantify the Semantic Consistency Gap.

Run the evaluation using the following command:

```bash
python evaluate_metrics.py 
```

3. Understanding the Output

The script will output a summary of results, including:
- **$\epsilon$ (Equivalence Coefficient)**: The degree of alignment between the Visual and Symbolic paths.
- **TEDS Score**: Measures HTML/TOO structural similarity via tree edit distance.
- **GriTS Score**: Evaluates the table as a 2D matrix, identifying precision and recall at the cell level.
