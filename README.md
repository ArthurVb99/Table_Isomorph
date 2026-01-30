# Project Structure: Two-Track LLM & Image Editing System

This project implements a dual-track system for AI-powered text and image processing using **LangChain** and **template-based prompts** with **multi-provider LLM support**.

## Quick Start

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

### 2. **Ollama** (Local Server - Free, Private)

```bash
# Install Ollama: https://ollama.ai

# Start Ollama server (default: http://localhost:11434)
ollama serve

# Pull a model
ollama pull mistral  # or llama2, neural-chat, etc.

# Configure in .env (optional, uses default if not set)
OLLAMA_BASE_URL=http://localhost:11434
# For remote server:
# OLLAMA_BASE_URL=http://134.184.22.126:11434

# Usage in Python
processor = LLMPromptProcessor(provider="ollama", model="mistral")
```

### 3. **HuggingFace** (Cloud API)

```bash
# Get API key from https://huggingface.co/settings/tokens

# Configure in .env
HUGGINGFACE_API_KEY=hf_...your_api_key...
# Optional: custom base URL
# HUGGINGFACE_BASE_URL=https://api-inference.huggingface.co/models

# Usage
processor = LLMPromptProcessor(
    provider="huggingface",
    model="meta-llama/Llama-2-7b-chat-hf"
)
```

### 4. **OpenRouter** (Multi-model aggregator)

```bash
# Get API key from https://openrouter.ai

# Configure in .env
OPENROUTER_API_KEY=sk-or-...your_api_key...
# Optional: custom base URL
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Usage
processor = LLMPromptProcessor(
    provider="openrouter",
    model="openai/gpt-3.5-turbo"
)
```

---

## Track 1: LLM Prompt Processor with Multi-Provider Support

**Directory:** `track_1_llm_prompt/`

Handles text-based prompt interactions with multiple LLM providers.

### Modules

- **llm_processor.py**: Multi-provider LLM interface
- **file_reader.py**: Read JSON, HTML, CSV files
- **data_transformer.py**: Row-column permutations and transformations

### Features

- Multi-model support (OpenAI, Ollama, HuggingFace, OpenRouter)
- Template-based prompt generation
- Batch prompt processing
- JSON/HTML/CSV file reading
- Row-column permutation and data transformation

### Basic LLM Usage

```python
from track_1_llm_prompt.llm_processor import LLMPromptProcessor

# Initialize with different providers
processor = LLMPromptProcessor(provider="openai", model="gpt-3.5-turbo")
# or
processor = LLMPromptProcessor(provider="ollama", model="mistral")

# Direct prompt
response = processor.process_prompt("What is AI?")

# Template-based
response = processor.process_template(
    "generic_question.txt",
    question="Explain quantum computing"
)

# Batch processing
responses = processor.process_batch_prompts([
    "What is AI?",
    "What is ML?",
    "What is DL?"
])

# List providers and templates
print(LLMPromptProcessor.get_available_providers())
print(processor.list_available_templates())
```

---

## Data Transformation & Table Permutations

### File Reading

```python
from track_1_llm_prompt.file_reader import DataFileReader
from track_1_llm_prompt.data_transformer import DataTransformer

# Read files (auto-detect format)
data = DataFileReader.read_file("data.json")
data = DataFileReader.read_html("table.html", table_index=0)
data = DataFileReader.read_csv("data.csv")

# Read all tables from HTML
all_tables = DataFileReader.read_html_all_tables("multi_table.html")
```

### Row-Column Permutations

```python
# Create transformer
transformer = DataTransformer(data)

# Swap rows (1-indexed for user convenience)
transformer.swap_rows_by_position(1, 2)  # Swap row 1 with row 2

# Swap columns
transformer.swap_columns("name", "age")  # Swap by column name
transformer.swap_columns_by_position(1, 2)  # Swap by position

# Apply multiple permutations
permutations = [
    {'type': 'row', 'first': 1, 'second': 2},
    {'type': 'column', 'first': 'name', 'second': 'age'},
    {'type': 'both', 
     'rows': [(0, 1), (2, 3)],
     'columns': [('col_a', 'col_b')]}
]
transformer.apply_row_column_permutations(permutations)

# Advanced operations
transformer.rotate_rows(2)  # Rotate rows down by 2
transformer.rotate_columns(1)  # Rotate columns
transformer.shuffle_rows(seed=42)  # Shuffle rows
transformer.shuffle_columns(seed=42)  # Shuffle columns

# Export
json_str = transformer.to_json()
dict_data = transformer.to_dict()
df = transformer.to_dataframe()
transformer.display()  # Print formatted table
```

### Complete Example

```python
from track_1_llm_prompt.file_reader import DataFileReader
from track_1_llm_prompt.data_transformer import DataTransformer
from track_1_llm_prompt.llm_processor import LLMPromptProcessor
import json

# 1. Read file
data = DataFileReader.read_file("employees.json")

# 2. Transform data
transformer = DataTransformer(data)
transformer.swap_rows_by_position(1, 2)
transformer.swap_columns("name", "dept")

# 3. Analyze with LLM
processor = LLMPromptProcessor(provider="openai")
response = processor.process_template(
    "table_permutation.txt",
    data=json.dumps(transformer.to_dict()),
    operation_type="both",
    num_permutations=2,
    constraints="Maintain employee hierarchy"
)

print(response)
```

---

## Track 2: Image Prompt Editor with Templates

**Directory:** `track_2_image_editing/`

Processes images based on text prompts or templates and outputs edited images.

### Features

- Prompt-based image editing
- Template-based editing instructions
- Multiple image operations
- Automatic output management

### Module: `image_editor.py`

**Basic Usage:**

```python
from track_2_image_editing.image_editor import ImagePromptEditor

# Initialize editor
editor = ImagePromptEditor(output_dir="output_images")

# Edit with text prompt
output_path = editor.process_with_prompt(
    image_path="input.jpg",
    prompt="Brighten the image and increase contrast",
    output_filename="brightened.png"
)
print(f"Edited image saved to: {output_path}")
```

**Template-Based Editing:**

```python
# List available templates
templates = editor.list_available_templates()

# Use basic edits template
output_path = editor.process_with_template(
    image_path="photo.jpg",
    template_name="basic_edits.txt",
    operations="sharpen, brighten",
    output_filename="enhanced.png"
)

# Use enhancement template with specific parameters
output_path = editor.process_with_template(
    image_path="landscape.jpg",
    template_name="enhancement.txt",
    brightness=1.2,
    contrast=1.3,
    saturation=1.5,
    apply="all",
    output_filename="landscape_enhanced.png"
)
```

---

## Supported Image Operations

| Operation | Method | Effect |
|---|---|---|
| Brighten | `adjust_brightness(factor > 1.0)` | Increase brightness |
| Darken | `adjust_brightness(factor < 1.0)` | Decrease brightness |
| Sharpen | `apply_sharpen(factor > 1.0)` | Enhance details |
| Blur | `apply_blur(radius)` | Apply gaussian blur |
| Grayscale | `apply_grayscale()` | Convert to black & white |
| Contrast | `adjust_contrast(factor > 1.0)` | Increase contrast |
| Saturate | `apply_saturation(factor > 1.0)` | Increase color saturation |
| Desaturate | `apply_saturation(factor < 1.0)` | Decrease color saturation |
| Resize | `resize_image(width, height)` | Scale image |

---

## Templates

### LLM Prompts (`templates/llm_prompts/`)

| Template | Purpose |
|---|---|
| generic_question.txt | General Q&A |
| text_analysis.txt | Analyze and extract information |
| code_help.txt | Code debugging and help |
| table_permutation.txt | Analyze tables and generate permutations |
| data_analysis.txt | Analyze file content and structure |
| sample_generation.txt | Generate sample data with transformations |

### Image Prompts (`templates/image_prompts/`)

| Template | Purpose |
|---|---|
| basic_edits.txt | Simple image operations |
| enhancement.txt | Brightness/contrast/saturation adjustments |

---

## Project Structure

```
TableStructureRecognitionProject/
├── track_1_llm_prompt/
│   ├── __init__.py
│   ├── llm_processor.py          # Multi-provider LLM interface
│   ├── file_reader.py             # JSON/HTML/CSV reading
│   └── data_transformer.py        # Row-column permutations
├── track_2_image_editing/
│   ├── __init__.py
│   └── image_editor.py            # Image editing
├── templates/
│   ├── llm_prompts/               # LLM prompt templates
│   │   ├── generic_question.txt
│   │   ├── text_analysis.txt
│   │   ├── code_help.txt
│   │   ├── table_permutation.txt
│   │   ├── data_analysis.txt
│   │   └── sample_generation.txt
│   └── image_prompts/             # Image editing templates
│       ├── basic_edits.txt
│       └── enhancement.txt
├── output_images/                 # Generated output images
├── examples.py                    # Comprehensive examples
├── requirements.txt
├── .env                           # Environment variables (create this)
├── README.md
└── .github/
    └── copilot-instructions.md
```

---

## Dependencies

- **langchain**: Multi-model LLM framework
- **langchain-openai**: OpenAI integration
- **openai**: OpenAI API client
- **requests**: HTTP client (for Ollama, HuggingFace, OpenRouter)
- **jinja2**: Template rendering
- **pillow**: Image processing
- **opencv-python**: Computer vision
- **python-dotenv**: Environment variables
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **beautifulsoup4**: HTML parsing
- **ollama**: Ollama client
- **huggingface-hub**: HuggingFace integration

---

## Usage Examples

### Example 1: OpenAI with Table Transformation

```python
from track_1_llm_prompt.llm_processor import LLMPromptProcessor
from track_1_llm_prompt.file_reader import DataFileReader
from track_1_llm_prompt.data_transformer import DataTransformer

processor = LLMPromptProcessor(provider="openai", model="gpt-3.5-turbo")
data = DataFileReader.read_json("data.json")
transformer = DataTransformer(data)
transformer.swap_rows_by_position(1, 2)
print(transformer.to_json())
```

### Example 2: Local Ollama

```python
processor = LLMPromptProcessor(
    provider="ollama",
    model="mistral",
    base_url="http://localhost:11434"
)
response = processor.process_prompt("What is data transformation?")
print(response)
```

### Example 3: HuggingFace

```python
processor = LLMPromptProcessor(
    provider="huggingface",
    model="meta-llama/Llama-2-7b-chat-hf"
)
response = processor.process_template(
    "generic_question.txt",
    question="Explain permutations"
)
```

### Example 4: Read HTML and Transform

```python
data = DataFileReader.read_html("employees.html")
transformer = DataTransformer(data)
transformer.shuffle_rows(seed=42)
print(transformer.to_json())
```

See `examples.py` for more complete examples.

---

## Troubleshooting

### ImportError: No module named 'langchain'

```bash
pip install -r requirements.txt
```

### Ollama: Connection refused

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, pull a model
ollama pull mistral
```

### API Key errors

1. Create `.env` file in project root
2. Add your API key: `OPENAI_API_KEY=your_key`
3. Never commit `.env` to git

### FileNotFoundError for templates

- Ensure `templates/llm_prompts/` and `templates/image_prompts/` exist
- Check template filename matches exactly

### HTML table not found

- Verify HTML file contains `<table>` element
- Try `read_html_all_tables()` to debug

---

## Advanced Features

### Custom LLM Models

```python
# Use different models with any provider
processor = LLMPromptProcessor(
    provider="openai",
    model="gpt-4",
    temperature=0.3  # More deterministic
)
```

### Create Custom Templates

1. Create `.txt` file in appropriate template directory
2. Use Jinja2 syntax: `{{ variable_name }}`
3. Use with `process_template()` method

### Chained Permutations

```python
transformer = DataTransformer(data)
transformer.swap_rows_by_position(1, 2) \
           .swap_columns("col_a", "col_b") \
           .rotate_rows(1) \
           .shuffle_columns(seed=42)
print(transformer.to_json())
```

---

## Future Enhancements

- [ ] Vision model integration (GPT-4V)
- [ ] Support for more local models (LLaMA, Mistral, etc.)
- [ ] Batch file processing pipeline
- [ ] REST API server
- [ ] Web UI dashboard
- [ ] Advanced table analysis with NLP
- [ ] Video processing support
- [ ] Multi-provider load balancing

---

## Notes

- **Never commit `.env`** to version control
- Both tracks operate independently
- Extend base classes for custom functionality
- Output images auto-saved to configured directory
- LangChain provides unified interface across models
- Local Ollama offers privacy without internet dependency

---

## Getting Help

1. Check `examples.py` for working code samples
2. Verify environment variables are set correctly
3. Ensure external services (Ollama, APIs) are running
4. Check template files exist in correct directories
5. Review error messages for specific provider issues
