"""
Batch Processing Script for Table Structure Recognition
Loads JSONL data, filters by token count, processes with LLM, validates, and outputs valid results.
"""

import os
import json
from track_1_llm_prompt.llm_processor import LLMPromptProcessor
from track_1_llm_prompt.validators import validate_llm_output, compare_cell_count


def main():
    # Get input path from environment variable
	input_path = os.getenv("PATH_INPUT_DATA_JSONL", "")
	provider = os.getenv("LLM_PROVIDER", "openai")
	model=os.getenv("model", "gpt-3.5-turbo")
	split_focus="train"
	# Maximum token limit (adjust as needed)
	max_tokens = 5000

	if not input_path:
		raise ValueError("PATH_INPUT environment variable not set")

	# Check if input file exists
	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input file not found: {input_path}")

	# Generate output path in same directory
	input_dir = os.path.dirname(input_path)
	input_filename = os.path.basename(input_path)
	output_filename = input_filename.replace('.jsonl', f'{split_focus}_processed.jsonl')
	output_path = os.path.join(input_dir, output_filename)

	# Initialize LLM processor (configure provider as needed)
	processor = LLMPromptProcessor(provider=provider, model=model)

	processed_count = 0
	valid_count = 0

	print(f"Processing {input_path}...")
	print(f"Output will be saved to: {output_path}")

	with open(input_path, 'r', encoding='utf-8') as in_file, \
			open(output_path, 'a', encoding='utf-8') as out_file:
		
		# also load the exiting imgids to avoid duplicates
		imgids_list= [json.loads(lin)['imgid'] if os.path.exists(output_path) else [] for lin in open(output_path, 'r', encoding='utf-8') ]

		for line_num, line in enumerate(in_file, 1):
			line = line.strip()
			if not line:
				continue

			try:
				# Parse JSON data
				data = json.loads(line)
				json_str = json.dumps(data)

				# process only specified split
				if data.get("split") != split_focus:
					print(f"Line {line_num}: Skipping, not in split '{split_focus}'")
					continue

				if len(imgids_list) > 0 and data.get("imgid") in imgids_list:
					print(f"Line {line_num}: Skipping, imgid {data.get('imgid')} already processed")
					continue

				# Check token count
				token_count = processor.count_input_tokens(json_str)
				if token_count > max_tokens:
					print(f"Line {line_num}: Token count {token_count} exceeds limit {max_tokens}, skipping")
					continue

				processed_count += 1
				print(f"Line {line_num}: Processing ({token_count} tokens)...")

				# Process with LLM
				llm_response = processor.process_template(
					"data_transform_transpose.txt",
					input_data=json_str
				)

				# Validate LLM output
				model, err = validate_llm_output(llm_response)
				if err:
					print(f"Line {line_num}: Validation failed: {err}")
					continue

				# Check cell count integrity
				ok = compare_cell_count(data, model.model_dump())
				if not ok:
					print(f"Line {line_num}: Cell count mismatch, skipping")
					continue

				# Write valid result to output
				output_data = model.model_dump()
				out_file.write(json.dumps(output_data) + '\n')
				valid_count += 1
				print(f"Line {line_num}: ✓ Valid result saved")

			except json.JSONDecodeError as e:
				print(f"Line {line_num}: Invalid JSON: {e}")
				continue
			except Exception as e:
				print(f"Line {line_num}: Processing error: {e}")
				continue

	print("Processing complete!")
	print(f"Total lines processed: {processed_count}")
	print(f"Valid results saved: {valid_count}")
	print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()