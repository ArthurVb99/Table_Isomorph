"""
TEDS Metric Calculator for Table Structure Recognition
Computes Tree Edit Distance Similarity between predicted and ground truth table structures.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from xml.etree import ElementTree as ET
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

load_dotenv()


class TEDSMetric:
    """
    Calculate TEDS (Tree Edit Distance Similarity) metric for table structures.
    TEDS = 1 - (TED / max(len(T1), len(T2)))
    where TED is the tree edit distance between two HTML trees.
    """

    def __init__(self, normalize: bool = True, ignore_case: bool = True):
        """
        Initialize TEDS metric calculator.

        Args:
            normalize (bool): Whether to normalize the HTML structure
            ignore_case (bool): Whether to ignore case when comparing tokens
        """
        self.normalize = normalize
        self.ignore_case = ignore_case

    def tokenize_html(self, html_str: str) -> List[str]:
        """
        Tokenize HTML string into individual tokens.

        Args:
            html_str (str): HTML structure as string

        Returns:
            List[str]: List of tokens
        """
        # Remove whitespace and split
        tokens = []
        current_token = ""

        for char in html_str:
            if char in '<>':
                if current_token.strip():
                    tokens.append(current_token.strip())
                if char == '<':
                    tokens.append('<')
                elif char == '>':
                    tokens.append('>')
                current_token = ""
            else:
                current_token += char

        if current_token.strip():
            tokens.append(current_token.strip())

        return tokens

    def normalize_token(self, token: str) -> str:
        """
        Normalize a token for comparison.

        Args:
            token (str): Token to normalize

        Returns:
            str: Normalized token
        """
        if self.ignore_case:
            token = token.lower()
        return token.strip()

    def tree_edit_distance(self, tokens1: List[str], tokens2: List[str]) -> int:
        """
        Calculate Tree Edit Distance using dynamic programming.
        Simplified version using sequence matching for HTML token sequences.

        Args:
            tokens1 (List[str]): First token sequence
            tokens2 (List[str]): Second token sequence

        Returns:
            int: Tree edit distance
        """
        m, n = len(tokens1), len(tokens2)

        # Initialize DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                token1 = self.normalize_token(tokens1[i - 1])
                token2 = self.normalize_token(tokens2[j - 1])

                if token1 == token2:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Cost of insertion, deletion, or substitution
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )

        return dp[m][n]

    def calculate_teds(self, pred_tokens: List[str], gt_tokens: List[str]) -> float:
        """
        Calculate TEDS score between predicted and ground truth tokens.

        Args:
            pred_tokens (List[str]): Predicted HTML tokens
            gt_tokens (List[str]): Ground truth HTML tokens

        Returns:
            float: TEDS score (0-1, higher is better)
        """
        ted = self.tree_edit_distance(pred_tokens, gt_tokens)
        max_len = max(len(pred_tokens), len(gt_tokens))

        teds_score = 1.0 - (ted / max_len)
        # return max(0.0, min(1.0, teds_score))  # Clamp to [0, 1]
        return teds_score

    def extract_structure_tokens(self, html_dict: Dict) -> List[str]:
        """
        Extract structure tokens from HTML dictionary.

        Args:
            html_dict (Dict): HTML dictionary with 'structure' key

        Returns:
            List[str]: List of structure tokens
        """
        if isinstance(html_dict, dict) and 'structure' in html_dict:
            structure = html_dict['structure']
            if isinstance(structure, dict) and 'tokens' in structure:
                return structure['tokens']
            elif isinstance(structure, list):
                return structure
        return []

    def match_by_imgid(self, pred_data: Dict, gt_data: Dict) -> Optional[Tuple[int, Dict, Dict]]:
        """
        Match prediction and ground truth by imgid.

        Args:
            pred_data (Dict): Predicted data (imgid -> entry)
            gt_data (Dict): Ground truth data (imgid -> entry)

        Returns:
            Optional[Tuple]: Matched entries with imgid
        """
        matched = []

        for imgid in pred_data:
            if imgid in gt_data:
                matched.append((imgid, pred_data[imgid], gt_data[imgid]))

        return matched
    
    # match by filename and split (give that oone file name is augmented with _transposed string)
    def match_by_filename_split(self, pred_data: Dict, gt_data: Dict) -> Optional[Tuple[int, Dict, Dict]]:
        """
        Match prediction and ground truth by imgid.

        Args:
            pred_data (Dict): Predicted data (imgid -> entry)
            gt_data (Dict): Ground truth data (imgid -> entry)
        Returns:
            Optional[Tuple]: Matched entries with imgid
        """
        matched = []
        for imgid in pred_data:
            pred_entry = pred_data[imgid]
            pred_filename = pred_entry.get('filename', '')
            pred_filename = pred_filename.replace('_transposed', '')
            pred_split = pred_entry.get('split', '')

            for gt_imgid, gt_entry in gt_data.items():
                gt_filename = gt_entry.get('filename', '')
                gt_filename = gt_filename.replace("_transposed",'')

                gt_split = gt_entry.get('split', '')

                # Check if filenames match after removing '_transposed'
                if (pred_filename== gt_filename) and (pred_split == gt_split):
                    matched.append((imgid, pred_entry, gt_entry))
                    break
            
        return matched
    
    def compare_single_table(self, pred_entry: Dict, gt_entry: Dict) -> Dict:
        """
        Compare a single predicted table with ground truth.

        Args:
            pred_entry (Dict): Predicted table entry
            gt_entry (Dict): Ground truth table entry

        Returns:
            Dict: Comparison results with TEDS score
        """
        result = {
            'imgid': pred_entry.get('imgid'),
            'teds': 0.0,
            'pred_tokens_count': 0,
            'gt_tokens_count': 0,
            'match': False
        }

        try:
            # Extract HTML structures
            pred_html = pred_entry.get('html', {})
            gt_html = gt_entry.get('html', {})

            # Get structure tokens
            pred_tokens = self.extract_structure_tokens(pred_html)
            gt_tokens = self.extract_structure_tokens(gt_html)

            result['pred_tokens_count'] = len(pred_tokens)
            result['gt_tokens_count'] = len(gt_tokens)

            # Calculate TEDS
            if pred_tokens and gt_tokens:
                teds_score = self.calculate_teds(pred_tokens, gt_tokens)
                result['teds'] = teds_score
                result['match'] = True

        except Exception as e:
            result['error'] = str(e)

        return result

    def load_jsonl(self, jsonl_path: str) -> Dict:
        """
        Load JSONL file into dictionary indexed by imgid.

        Args:
            jsonl_path (str): Path to JSONL file

        Returns:
            Dict: Dictionary with imgid as key
        """
        data = {}

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        imgid = entry.get('imgid')
                        if imgid:
                            data[imgid] = entry
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Line {line_num} - JSON decode error: {e}")

        except FileNotFoundError:
            print(f"Error: File not found: {jsonl_path}")
            return {}

        print(f"Loaded {len(data)} entries from {jsonl_path}")
        return data

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """
        Compute overall metrics from comparison results.

        Args:
            results (List[Dict]): List of comparison results

        Returns:
            Dict: Aggregated metrics
        """
        if not results:
            return {
                'total': 0,
                'matched': 0,
                'mean_teds': 0.0,
                'min_teds': 0.0,
                'max_teds': 1.0,
                'std_teds': 0.0
            }

        teds_scores = [r['teds'] for r in results if r.get('match')]

        if not teds_scores:
            return {
                'total': len(results),
                'matched': 0,
                'mean_teds': 0.0,
                'min_teds': 0.0,
                'max_teds': 1.0,
                'std_teds': 0.0
            }

        mean_teds = sum(teds_scores) / len(teds_scores)
        min_teds = min(teds_scores)
        max_teds = max(teds_scores)

        # Calculate standard deviation
        variance = sum((x - mean_teds) ** 2 for x in teds_scores) / len(teds_scores)
        std_teds = variance ** 0.5

        return {
            'total': len(results),
            'matched': len(teds_scores),
            'mean_teds': round(mean_teds, 4),
            'min_teds': round(min_teds, 4),
            'max_teds': round(max_teds, 4),
            'std_teds': round(std_teds, 4)
        }

    def compare_jsonl_files(self, pred_jsonl: str, gt_jsonl: str, output_path: Optional[str] = None,
                           max_workers: int = 4, matching_by_imgid: bool = True) -> Dict:
        """
        Compare two JSONL files and compute TEDS metrics.

        Args:
            pred_jsonl (str): Path to predicted JSONL file
            gt_jsonl (str): Path to ground truth JSONL file
            output_path (Optional[str]): Path to save detailed results
            max_workers (int): Number of parallel threads

        Returns:
            Dict: Aggregated metrics and results
        """
        print(f"Loading ground truth from: {gt_jsonl}")
        gt_data = self.load_jsonl(gt_jsonl)

        print(f"Loading predictions from: {pred_jsonl}")
        pred_data = self.load_jsonl(pred_jsonl)

        # # Match by imgid
        # print("\nMatching entries by imgid...")
        if matching_by_imgid:
            matched_pairs = self.match_by_imgid(pred_data, gt_data)
        else:
            print("\nMatching entries by filename and split...")
            matched_pairs = self.match_by_filename_split(pred_data, gt_data)

        print(f"Matched {len(matched_pairs)} entries out of {len(pred_data)} predictions")

        # Process comparisons in parallel
        print(f"\nComputing TEDS metrics with {max_workers} threads...")
        results = []
        output_lock = Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.compare_single_table, pred, gt): imgid
                for imgid, pred, gt in matched_pairs
            }

            for future in as_completed(futures):
                imgid = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.get('match'):
                        with output_lock:
                            print(f"✓ imgid {imgid}: TEDS = {result['teds']:.4f}")
                    else:
                        with output_lock:
                            print(f"✗ imgid {imgid}: Error - {result.get('error', 'Unknown error')}")

                except Exception as e:
                    with output_lock:
                        print(f"✗ imgid {imgid}: Exception - {str(e)}")

        # Compute aggregated metrics
        metrics = self.compute_metrics(results)

        # Save results if output path specified
        if output_path:
            self._save_results(results, metrics, output_path)

        return {
            'metrics': metrics,
            'results': results
        }

    def _save_results(self, results: List[Dict], metrics: Dict, output_path: str):
        """Save detailed results to JSONL file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write summary
                f.write(json.dumps({'summary': metrics}) + '\n')

                # Write individual results
                for result in results:
                    f.write(json.dumps(result) + '\n')

            print(f"\n✓ Results saved to: {output_path}")

        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Main function - compare JSONL files and compute TEDS metrics."""

    pred_jsonl = os.getenv("PATH_PRED_JSONL", "")
    gt_jsonl = os.getenv("PATH_GT_JSONL", "")
    output_path = os.getenv("PATH_OUTPUT_RESULTS", None)
    max_threads = int(os.getenv("MAX_THREADS", "4"))
    matching_by_imgid = bool(int(os.getenv("MATCHING_BY_IMGID", "1")))

    if not pred_jsonl or not gt_jsonl:
        raise ValueError("PATH_PRED_JSONL and PATH_GT_JSONL environment variables must be set")

    # Initialize TEDS calculator
    teds = TEDSMetric(normalize=True, ignore_case=True)

    # Compare files
    result = teds.compare_jsonl_files(
        pred_jsonl=pred_jsonl,
        gt_jsonl=gt_jsonl,
        output_path=output_path,
        max_workers=max_threads,
        matching_by_imgid=matching_by_imgid
    )

    # Print summary
    print("\n" + "=" * 80)
    print("TEDS METRIC SUMMARY")
    print("=" * 80)
    metrics = result['metrics']
    print(f"Total entries compared: {metrics['total']}")
    print(f"Matched entries: {metrics['matched']}")
    print(f"Mean TEDS: {metrics['mean_teds']}")
    print(f"Min TEDS: {metrics['min_teds']}")
    print(f"Max TEDS: {metrics['max_teds']}")
    print(f"Std Dev TEDS: {metrics['std_teds']}")
    print("=" * 80)


if __name__ == "__main__":
    main()