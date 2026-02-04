"""
Comprehensive Table Evaluation Script
Evaluates table structure predictions using TEDS and GriTS metrics from table-metrics library.
Supports matching by imgid or filename+split.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv


from table_metrics import teds_score, grits_top_score, grits_con_score, grits_loc_score


load_dotenv()


def json_to_html(data: Dict) -> str:
    """
    Convert JSON table structure to HTML string.
    
    Args:
        data (Dict): JSON structure containing 'html' with 'cells' and 'structure'
        
    Returns:
        str: Complete HTML table string
    """
    html_dict = data.get("html", {})
    cells = html_dict.get("cells", [])
    structure = html_dict.get("structure", {})
    structure_tokens = structure.get("tokens", []) if isinstance(structure, dict) else structure
    
    if not structure_tokens:
        return ""
    
    # Build cell content list
    cell_contents = []
    for cell in cells:
        tokens = cell.get("tokens", [])
        # Join tokens with space and strip
        content = " ".join(tokens).strip()
        cell_contents.append(content)
    
    # Merge structure with content
    html_parts = []
    cell_index = 0
    
    for token in structure_tokens:
        if token == "<td>" or token == "<th>":
            # Add opening tag
            html_parts.append(token)
            
            # ALWAYS add cell content (even if empty) and increment index
            if cell_index < len(cell_contents):
                html_parts.append(cell_contents[cell_index])
            else:
                html_parts.append("")  # Empty cell fallback
            
            cell_index += 1  # Always increment
        else:
            # Add structure token as-is
            html_parts.append(token)
    
    # Wrap in table tag and join
    html_string = "<table>" + "".join(html_parts) + "</table>"
    
    return html_string


class TableMetricsEvaluator:
    """
    Evaluate table structure predictions using TEDS and GriTS metrics.
    """

    def __init__(self, structure_only: bool = False, ignored_nodes: Optional[List[str]] = None):
        """
        Initialize evaluator.

        Args:
            structure_only (bool): If True, only compare structure (ignore content) for TEDS
            ignored_nodes (Optional[List[str]]): HTML tags to ignore during TEDS comparison
        """
        self.structure_only = structure_only
        self.ignored_nodes = ignored_nodes or []

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
                        if imgid is not None:
                            data[imgid] = entry
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Line {line_num} - JSON decode error: {e}")

        except FileNotFoundError:
            print(f"Error: File not found: {jsonl_path}")
            return {}

        print(f"Loaded {len(data)} entries from {jsonl_path}")
        return data

    def match_by_imgid(self, pred_data: Dict, gt_data: Dict) -> List[Tuple]:
        """
        Match prediction and ground truth by imgid.

        Args:
            pred_data (Dict): Predicted data (imgid -> entry)
            gt_data (Dict): Ground truth data (imgid -> entry)

        Returns:
            List[Tuple]: Matched entries with imgid
        """
        matched = []

        for imgid in pred_data:
            if imgid in gt_data:
                matched.append((imgid, pred_data[imgid], gt_data[imgid]))

        return matched

    def match_by_filename(self, pred_data: Dict, gt_data: Dict) -> List[Tuple]:
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

    def evaluate_single_table(self, pred_entry: Dict, gt_entry: Dict) -> Dict:
        """
        Evaluate a single predicted table against ground truth using TEDS and GriTS.

        Args:
            pred_entry (Dict): Predicted table entry
            gt_entry (Dict): Ground truth table entry

        Returns:
            Dict: Evaluation results with all metrics
        """
        result = {
            'imgid': pred_entry.get('imgid'),
            'filename': pred_entry.get('filename'),
            'teds': 0.0,
            'teds_struct': 0.0,
            'grits_top': 0.0,
            'grits_con': 0.0,
            'grits_loc': 0.0,
            'match': False
        }

        try:
            # Convert JSON to HTML
            pred_html = json_to_html(pred_entry)
            gt_html = json_to_html(gt_entry)

            if not pred_html or not gt_html:
                result['error'] = 'Failed to convert JSON to HTML'
                return result

            # Calculate TEDS (with content)
            teds_full = teds_score(
                y_true=gt_html,
                y_pred=pred_html,
                structure_only=False,
                ignored_nodes=self.ignored_nodes
            )

            # Calculate TEDS (structure only)
            teds_struct = teds_score(
                y_true=gt_html,
                y_pred=pred_html,
                structure_only=True,
                ignored_nodes=self.ignored_nodes
            )

            # Calculate GriTS metrics
            grits_top = grits_top_score(gt_html, pred_html)
            grits_con = grits_con_score(gt_html, pred_html)
            grits_loc = 0.00

            result['teds'] = round(teds_full, 4)
            result['teds_struct'] = round(teds_struct, 4)
            result['grits_top'] = round(grits_top, 4)
            result['grits_con'] = round(grits_con, 4)
            result['grits_loc'] = round(grits_loc, 4)
            result['match'] = True

        except Exception as e:
            result['error'] = str(e)

        return result

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """
        Compute overall metrics from evaluation results.

        Args:
            results (List[Dict]): List of evaluation results

        Returns:
            Dict: Aggregated metrics
        """
        if not results:
            return {
                'total': 0,
                'matched': 0,
                'mean_teds': 0.0,
                'mean_teds_struct': 0.0,
                'mean_grits_top': 0.0,
                'mean_grits_con': 0.0,
                'mean_grits_loc': 0.0,
                'min_teds': 0.0,
                'max_teds': 1.0,
                'std_teds': 0.0,
                'std_grits_top': 0.0,
                'std_grits_con': 0.0,
                'std_grits_loc': 0.0
            }

        matched_results = [r for r in results if r.get('match')]

        if not matched_results:
            return {
                'total': len(results),
                'matched': 0,
                'mean_teds': 0.0,
                'mean_teds_struct': 0.0,
                'mean_grits_top': 0.0,
                'mean_grits_con': 0.0,
                'mean_grits_loc': 0.0,
                'min_teds': 0.0,
                'max_teds': 1.0,
                'std_teds': 0.0,
                'std_grits_top': 0.0,
                'std_grits_con': 0.0,
                'std_grits_loc': 0.0
            }

        teds_scores = [r['teds'] for r in matched_results]
        teds_struct_scores = [r['teds_struct'] for r in matched_results]
        grits_top_scores = [r['grits_top'] for r in matched_results]
        grits_con_scores = [r['grits_con'] for r in matched_results]
        grits_loc_scores = [r['grits_loc'] for r in matched_results]

        mean_teds = sum(teds_scores) / len(teds_scores)
        mean_teds_struct = sum(teds_struct_scores) / len(teds_struct_scores)
        mean_grits_top = sum(grits_top_scores) / len(grits_top_scores)
        mean_grits_con = sum(grits_con_scores) / len(grits_con_scores)
        mean_grits_loc = sum(grits_loc_scores) / len(grits_loc_scores)
        
        min_teds = min(teds_scores)
        max_teds = max(teds_scores)
        min_grits_top = min(grits_top_scores)
        max_grits_top = max(grits_top_scores)
        min_grits_con = min(grits_con_scores)
        max_grits_con = max(grits_con_scores)
        min_grits_loc = min(grits_loc_scores)
        max_grits_loc = max(grits_loc_scores)

        # Calculate standard deviations
        variance_teds = sum((x - mean_teds) ** 2 for x in teds_scores) / len(teds_scores)
        std_teds = variance_teds ** 0.5
        
        variance_grits_top = sum((x - mean_grits_top) ** 2 for x in grits_top_scores) / len(grits_top_scores)
        std_grits_top = variance_grits_top ** 0.5
        
        variance_grits_con = sum((x - mean_grits_con) ** 2 for x in grits_con_scores) / len(grits_con_scores)
        std_grits_con = variance_grits_con ** 0.5

        variance_grits_loc = sum((x - mean_grits_loc) ** 2 for x in grits_loc_scores) / len(grits_loc_scores)
        std_grits_loc = variance_grits_loc ** 0.5

        return {
            'total': len(results),
            'matched': len(matched_results),
            'mean_teds': round(mean_teds, 4),
            'mean_teds_struct': round(mean_teds_struct, 4),
            'mean_grits_top': round(mean_grits_top, 4),
            'mean_grits_con': round(mean_grits_con, 4),
            'mean_grits_loc': round(mean_grits_loc, 4),
            'min_teds': round(min_teds, 4),
            'max_teds': round(max_teds, 4),
            'min_grits_top': round(min_grits_top, 4),
            'max_grits_top': round(max_grits_top, 4),
            'min_grits_con': round(min_grits_con, 4),
            'max_grits_con': round(max_grits_con, 4),
            'min_grits_loc': round(min_grits_loc, 4),
            'max_grits_loc': round(max_grits_loc, 4),
            'std_teds': round(std_teds, 4),
            'std_grits_top': round(std_grits_top, 4),
            'std_grits_con': round(std_grits_con, 4),
            'std_grits_loc': round(std_grits_loc, 4)
        }

    def evaluate_jsonl_files(
        self,
        pred_jsonl: str,
        gt_jsonl: str,
        output_path: Optional[str] = None,
        max_workers: int = 4,
        matching_by_imgid: bool = True
    ) -> Dict:
        """
        Evaluate predictions against ground truth from JSONL files.

        Args:
            pred_jsonl (str): Path to predicted JSONL file
            gt_jsonl (str): Path to ground truth JSONL file
            output_path (Optional[str]): Path to save detailed results
            max_workers (int): Number of parallel threads
            matching_by_imgid (bool): Whether to match by imgid (True) or filename (False)

        Returns:
            Dict: Aggregated metrics and results
        """
        print(f"Loading ground truth from: {gt_jsonl}")
        gt_data = self.load_jsonl(gt_jsonl)

        print(f"Loading predictions from: {pred_jsonl}")
        pred_data = self.load_jsonl(pred_jsonl)

        # Match by strategy
        if matching_by_imgid:
            print("\nMatching entries by imgid...")
            matched_pairs = self.match_by_imgid(pred_data, gt_data)
        else:
            print("\nMatching entries by filename and split...")
            matched_pairs = self.match_by_filename(pred_data, gt_data)
            
        print(f"Matched {len(matched_pairs)} entries out of {len(pred_data)} predictions")

        # Process evaluations in parallel
        print(f"\nComputing metrics with {max_workers} threads...")
        results = []
        output_lock = Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_single_table, pred, gt): imgid
                for imgid, pred, gt in matched_pairs
            }

            for future in as_completed(futures):
                imgid = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.get('match'):
                        with output_lock:
                            print(f"✓ imgid {imgid}: "
                                  f"TEDS={result['teds']:.4f}, "
                                  f"TEDS-Struct={result['teds_struct']:.4f}, "
                                  f"GriTS_Top={result['grits_top']:.4f}, "
                                  f"GriTS_Con={result['grits_con']:.4f}, "
                                  f"GriTS_Loc={result['grits_loc']:.4f}")
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
    """Main function - evaluate predictions against ground truth."""

    pred_jsonl = os.getenv("PATH_PRED_JSONL", "")
    gt_jsonl = os.getenv("PATH_GT_JSONL", "")
    output_path = os.getenv("PATH_OUTPUT_RESULTS", None)
    max_threads = int(os.getenv("MAX_THREADS", "4"))
    matching_by_imgid = bool(int(os.getenv("MATCHING_BY_IMGID", "0")))  
    structure_only = bool(int(os.getenv("STRUCTURE_ONLY", "0")))

    if not pred_jsonl or not gt_jsonl:
        raise ValueError("PATH_PRED_JSONL and PATH_GT_JSONL environment variables must be set")

    # Initialize evaluator
    evaluator = TableMetricsEvaluator(
        structure_only=structure_only,
        ignored_nodes=[]
    )

    # Evaluate files
    result = evaluator.evaluate_jsonl_files(
        pred_jsonl=pred_jsonl,
        gt_jsonl=gt_jsonl,
        output_path=output_path,
        max_workers=max_threads,
        matching_by_imgid=matching_by_imgid
    )

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)
    metrics = result['metrics']
    print(f"Total entries compared: {metrics['total']}")
    print(f"Matched entries: {metrics['matched']}")
    print()
    print("TEDS Metrics:")
    print(f"  Mean TEDS (with content): {metrics['mean_teds']}")
    print(f"  Mean TEDS (structure only): {metrics['mean_teds_struct']}")
    print(f"  Min TEDS: {metrics['min_teds']}")
    print(f"  Max TEDS: {metrics['max_teds']}")
    print(f"  Std Dev TEDS: {metrics['std_teds']}")
    print()
    print("GriTS Metrics:")
    print(f"  Mean GriTS_Top: {metrics['mean_grits_top']}")
    print(f"  Mean GriTS_Con: {metrics['mean_grits_con']}")
    print(f"  Mean GriTS_Loc: {metrics['mean_grits_loc']}")
    print(f"  Min GriTS_Top: {metrics['min_grits_top']}")
    print(f"  Max GriTS_Top: {metrics['max_grits_top']}")
    print(f"  Min GriTS_Con: {metrics['min_grits_con']}")
    print(f"  Max GriTS_Con: {metrics['max_grits_con']}")
    print(f"  Min GriTS_Loc: {metrics['min_grits_loc']}")
    print(f"  Max GriTS_Loc: {metrics['max_grits_loc']}")
    print(f"  Std Dev GriTS_Top: {metrics['std_grits_top']}")
    print(f"  Std Dev GriTS_Con: {metrics['std_grits_con']}")
    print(f"  Std Dev GriTS_Loc: {metrics['std_grits_loc']}")
    print("=" * 80)


if __name__ == "__main__":
    main()