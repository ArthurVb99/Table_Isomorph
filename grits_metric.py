"""
GriTS Metric Calculator for Table Structure Recognition
Adapted from the official Table Transformer implementation:
https://github.com/microsoft/table-transformer/blob/main/src/grits.py
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def _flatten_structure_tokens(structure_tokens: List[str]) -> str:
    """Convert structure tokens to an HTML string."""
    if not structure_tokens:
        return ""
    return "".join(structure_tokens)


def _safe_text(tokens: List[str]) -> str:
    """Join cell tokens to a normalized text string."""
    return " ".join(tokens).strip().lower()


def _extract_cells(entry: Dict) -> List[Dict]:
    """Extract cell list from JSONL entry."""
    html = entry.get("html", {})
    return html.get("cells", []) if isinstance(html, dict) else []


def _extract_structure_tokens(entry: Dict) -> List[str]:
    """Extract structure tokens list from JSONL entry."""
    html = entry.get("html", {})
    if isinstance(html, dict):
        structure = html.get("structure", {})
        if isinstance(structure, dict):
            return structure.get("tokens", [])
        if isinstance(structure, list):
            return structure
    return []


def _cell_texts(cells: List[Dict]) -> List[str]:
    """Get normalized cell text list."""
    return [_safe_text(c.get("tokens", [])) for c in cells]


def _levenshtein(a: List[str], b: List[str]) -> int:
    """Levenshtein distance for token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)

    m, n = len(a), len(b)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,       # deletion
                dp[j - 1] + 1,   # insertion
                prev + cost      # substitution
            )
            prev = cur

    return dp[n]


def _similarity(a: List[str], b: List[str]) -> float:
    """Normalized similarity from Levenshtein distance."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = _levenshtein(a, b)
    denom = max(len(a), len(b))
    return max(0.0, 1.0 - (dist / denom))


def _grits_from_tokens(pred_structure: List[str], gt_structure: List[str],
                       pred_cells: List[Dict], gt_cells: List[Dict]) -> Dict:
    """
    Compute GriTS metrics: structure (GriTS_S), content (GriTS_C), and overall (GriTS).
    """
    # Structure similarity
    s_score = _similarity(pred_structure, gt_structure)

    # Content similarity: compare normalized cell text sequences
    pred_texts = _cell_texts(pred_cells)
    gt_texts = _cell_texts(gt_cells)
    c_score = _similarity(pred_texts, gt_texts)

    # Overall score (official uses product-like weighting; here harmonic mean)
    if s_score + c_score == 0:
        g_score = 0.0
    else:
        g_score = 2 * s_score * c_score / (s_score + c_score)

    return {
        "grits_s": round(s_score, 4),
        "grits_c": round(c_score, 4),
        "grits": round(g_score, 4),
    }


def _load_jsonl(path: str) -> Dict:
    """Load JSONL into dict keyed by imgid."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                imgid = entry.get("imgid")
                if imgid is not None:
                    data[imgid] = entry
            except json.JSONDecodeError:
                continue
    return data


def _match_by_imgid(pred_data: Dict, gt_data: Dict) -> List[Tuple[int, Dict, Dict]]:
    matched = []
    for imgid, pred_entry in pred_data.items():
        if imgid in gt_data:
            matched.append((imgid, pred_entry, gt_data[imgid]))
    return matched


def compare_single_table(pred_entry: Dict, gt_entry: Dict) -> Dict:
    """Compare one predicted table vs ground truth using GriTS."""
    pred_structure = _extract_structure_tokens(pred_entry)
    gt_structure = _extract_structure_tokens(gt_entry)

    pred_cells = _extract_cells(pred_entry)
    gt_cells = _extract_cells(gt_entry)

    scores = _grits_from_tokens(pred_structure, gt_structure, pred_cells, gt_cells)

    return {
        "imgid": pred_entry.get("imgid"),
        "filename": pred_entry.get("filename"),
        **scores,
        "match": True
    }


def compute_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {"total": 0, "mean_grits": 0.0, "mean_grits_s": 0.0, "mean_grits_c": 0.0}

    mean_grits = sum(r["grits"] for r in results) / len(results)
    mean_grits_s = sum(r["grits_s"] for r in results) / len(results)
    mean_grits_c = sum(r["grits_c"] for r in results) / len(results)

    return {
        "total": len(results),
        "mean_grits": round(mean_grits, 4),
        "mean_grits_s": round(mean_grits_s, 4),
        "mean_grits_c": round(mean_grits_c, 4),
    }


def compare_jsonl_files(pred_jsonl: str, gt_jsonl: str,
                        output_path: Optional[str] = None,
                        max_workers: int = 4) -> Dict:
    """Compare two JSONL files and compute GriTS metrics in parallel."""
    pred_data = _load_jsonl(pred_jsonl)
    gt_data = _load_jsonl(gt_jsonl)

    matched = _match_by_imgid(pred_data, gt_data)

    results = []
    output_lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compare_single_table, pred, gt): imgid
            for imgid, pred, gt in matched
        }

        for future in as_completed(futures):
            imgid = futures[future]
            try:
                res = future.result()
                results.append(res)
                with output_lock:
                    print(f"✓ imgid {imgid}: GriTS={res['grits']:.4f} "
                          f"(S={res['grits_s']:.4f}, C={res['grits_c']:.4f})")
            except Exception as e:
                with output_lock:
                    print(f"✗ imgid {imgid}: {e}")

    metrics = compute_metrics(results)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"summary": metrics}) + "\n")
            for r in results:
                f.write(json.dumps(r) + "\n")

    return {"metrics": metrics, "results": results}


def main():
    pred_jsonl = os.getenv("PATH_PRED_JSONL", "")
    gt_jsonl = os.getenv("PATH_GT_JSONL", "")
    output_path = os.getenv("PATH_OUTPUT_RESULTS", None)
    max_threads = int(os.getenv("MAX_THREADS", "4"))

    if not pred_jsonl or not gt_jsonl:
        raise ValueError("PATH_PRED_JSONL and PATH_GT_JSONL must be set.")

    result = compare_jsonl_files(
        pred_jsonl=pred_jsonl,
        gt_jsonl=gt_jsonl,
        output_path=output_path,
        max_workers=max_threads
    )

    print("\n" + "=" * 80)
    print("GRITS METRIC SUMMARY")
    print("=" * 80)
    print(result["metrics"])
    print("=" * 80)


if __name__ == "__main__":
    main()