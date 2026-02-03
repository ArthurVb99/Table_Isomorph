from dataclasses import dataclass
from rapidfuzz.distance import LCSseq
from lxml import html, etree
from collections import defaultdict
from typing import Self

import numpy as np


@dataclass(frozen=True, slots=True)
class _Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def __init__(self, bbox: Self | tuple | list = (0, 0, 0, 0)):
        if isinstance(bbox, _Rect):
            object.__setattr__(self, "x0", bbox.x0)
            object.__setattr__(self, "y0", bbox.y0)
            object.__setattr__(self, "x1", bbox.x1)
            object.__setattr__(self, "y1", bbox.y1)
        else:
            object.__setattr__(self, "x0", bbox[0])
            object.__setattr__(self, "y0", bbox[1])
            object.__setattr__(self, "x1", bbox[2])
            object.__setattr__(self, "y1", bbox[3])

    def get_area(self) -> float:
        width = max(0, self.x1 - self.x0)
        height = max(0, self.y1 - self.y0)

        return width * height

    def intersect(self, other: Self | tuple | list) -> Self:
        if not isinstance(other, _Rect):
            other = _Rect(other)

        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)

        if x1 < x0 or y1 < y0:
            return _Rect([0, 0, 0, 0])
        else:
            return _Rect([x0, y0, x1, y1])

    def include_rect(self, other: Self | tuple | list) -> Self:
        if not isinstance(other, _Rect):
            other = _Rect(other)

        x0 = min(self.x0, other.x0)
        y0 = min(self.y0, other.y0)
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)

        return _Rect([x0, y0, x1, y1])


def _compute_fscore(num_true_positives, num_true, num_positives):
    if num_positives:
        precision = num_true_positives / num_positives
    else:
        precision = 1

    if num_true:
        recall = num_true_positives / num_true
    else:
        recall = 1

    if precision + recall:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0

    return fscore, precision, recall


def _initialize_DP(sequence1_length, sequence2_length, need_pointers: bool = False):
    scores = [[0] * (sequence2_length + 1) for _ in range(sequence1_length + 1)]

    if need_pointers:
        pointers = [[0] * (sequence2_length + 1) for _ in range(sequence1_length + 1)]
        for seq1_idx in range(1, sequence1_length + 1):
            pointers[seq1_idx][0] = -1

        for seq2_idx in range(1, sequence2_length + 1):
            pointers[0][seq2_idx] = 1

        return scores, pointers
    else:
        return scores, None


def _traceback(pointers):
    seq1_idx = len(pointers) - 1
    seq2_idx = len(pointers[0]) - 1
    aligned_sequence1_indices = []
    aligned_sequence2_indices = []
    while not (seq1_idx == 0 and seq2_idx == 0):
        if pointers[seq1_idx][seq2_idx] == -1:
            seq1_idx -= 1
        elif pointers[seq1_idx][seq2_idx] == 1:
            seq2_idx -= 1
        else:
            seq1_idx -= 1
            seq2_idx -= 1
            aligned_sequence1_indices.append(seq1_idx)
            aligned_sequence2_indices.append(seq2_idx)

    aligned_sequence1_indices = aligned_sequence1_indices[::-1]
    aligned_sequence2_indices = aligned_sequence2_indices[::-1]

    return aligned_sequence1_indices, aligned_sequence2_indices


def _align_1d(
    sequence1, sequence2, reward_lookup, return_alignment=False, is_transposed=False
):
    sequence1_length = len(sequence1)
    sequence2_length = len(sequence2)

    scores, pointers = _initialize_DP(
        sequence1_length, sequence2_length, need_pointers=return_alignment
    )

    for seq1_idx in range(1, sequence1_length + 1):
        for seq2_idx in range(1, sequence2_length + 1):
            trow, tcol = sequence1[seq1_idx - 1]
            prow, pcol = sequence2[seq2_idx - 1]
            reward = reward_lookup(trow, tcol, prow, pcol, is_transposed=is_transposed)

            diag_score = scores[seq1_idx - 1][seq2_idx - 1] + reward
            skip_seq2_score = scores[seq1_idx][seq2_idx - 1]
            skip_seq1_score = scores[seq1_idx - 1][seq2_idx]

            max_score = max(diag_score, skip_seq1_score, skip_seq2_score)
            scores[seq1_idx][seq2_idx] = max_score

            if return_alignment:
                if diag_score == max_score:
                    pointers[seq1_idx][seq2_idx] = 0
                elif skip_seq1_score == max_score:
                    pointers[seq1_idx][seq2_idx] = -1
                else:
                    pointers[seq1_idx][seq2_idx] = 1

    score = scores[-1][-1]

    if return_alignment:
        sequence1_indices, sequence2_indices = _traceback(pointers)

        return sequence1_indices, sequence2_indices, score
    else:
        return score


def _align_2d_outer(true_shape, pred_shape, reward_lookup, is_transposed=False):
    scores, pointers = _initialize_DP(true_shape[0], pred_shape[0], need_pointers=True)

    for row_idx in range(1, true_shape[0] + 1):
        for col_idx in range(1, pred_shape[0] + 1):
            reward = _align_1d(
                [(row_idx - 1, tcol) for tcol in range(true_shape[1])],
                [(col_idx - 1, prow) for prow in range(pred_shape[1])],
                reward_lookup,
                is_transposed=is_transposed,
            )
            diag_score = scores[row_idx - 1][col_idx - 1] + reward
            same_row_score = scores[row_idx][col_idx - 1]
            same_col_score = scores[row_idx - 1][col_idx]

            max_score = max(diag_score, same_col_score, same_row_score)
            scores[row_idx][col_idx] = max_score
            if diag_score == max_score:
                pointers[row_idx][col_idx] = 0
            elif same_col_score == max_score:
                pointers[row_idx][col_idx] = -1
            else:
                pointers[row_idx][col_idx] = 1

    score = scores[-1][-1]

    aligned_true_indices, aligned_pred_indices = _traceback(pointers)

    return aligned_true_indices, aligned_pred_indices, score


def _factored_2dmss(true_cell_grid, pred_cell_grid, reward_function):
    reward_cache = {}

    def get_reward(trow: int, tcol: int, prow: int, pcol: int, is_transposed=False):
        if is_transposed:
            key = (tcol, trow, pcol, prow)
        else:
            key = (trow, tcol, prow, pcol)
        if key not in reward_cache:
            reward_cache[key] = reward_function(
                true_cell_grid[trow, tcol], pred_cell_grid[prow, pcol]
            )

        return reward_cache[key]

    true_row_nums, pred_row_nums, row_pos_match_score = _align_2d_outer(
        true_cell_grid.shape[:2], pred_cell_grid.shape[:2], get_reward
    )

    true_column_nums, pred_column_nums, col_pos_match_score = _align_2d_outer(
        true_cell_grid.shape[:2][::-1],
        pred_cell_grid.shape[:2][::-1],
        get_reward,
        is_transposed=True,
    )

    num_pos = pred_cell_grid.shape[0] * pred_cell_grid.shape[1]
    num_true = true_cell_grid.shape[0] * true_cell_grid.shape[1]

    pos_match_score_upper_bound = min(row_pos_match_score, col_pos_match_score)
    upper_bound_score, _, _ = _compute_fscore(
        pos_match_score_upper_bound, num_pos, num_true
    )

    positive_match_score = 0
    for true_row_num, pred_row_num in zip(true_row_nums, pred_row_nums):
        for true_column_num, pred_column_num in zip(true_column_nums, pred_column_nums):
            positive_match_score += reward_cache[
                (true_row_num, true_column_num, pred_row_num, pred_column_num)
            ]

    fscore, precision, recall = _compute_fscore(positive_match_score, num_true, num_pos)

    return fscore, precision, recall, upper_bound_score


def _lcs_similarity(string1, string2):
    return LCSseq.normalized_similarity(string1, string2)


def _iou(bbox1, bbox2):
    first_rect = _Rect(bbox1)
    second_rect = _Rect(bbox2)
    intersection = first_rect.intersect(second_rect)
    union = first_rect.include_rect(second_rect)

    union_area = union.get_area()
    if union_area:
        return intersection.get_area() / union_area

    return 0


def _cells_to_grid(cells, key="bbox"):
    if len(cells) == 0:
        return [[]]
    num_rows = max([max(cell["row_nums"]) for cell in cells]) + 1
    num_columns = max([max(cell["column_nums"]) for cell in cells]) + 1
    cell_grid = [[0] * num_columns for _ in range(num_rows)]
    for cell in cells:
        for row_num in cell["row_nums"]:
            for column_num in cell["column_nums"]:
                cell_grid[row_num][column_num] = cell[key]

    return cell_grid


def _cells_to_relspan_grid(cells):
    if len(cells) == 0:
        return [[]]

    num_rows = max([max(cell["row_nums"]) for cell in cells]) + 1
    num_columns = max([max(cell["column_nums"]) for cell in cells]) + 1
    cell_grid = [[0] * num_columns for _ in range(num_rows)]
    for cell in cells:
        min_row_num = min(cell["row_nums"])
        min_column_num = min(cell["column_nums"])
        max_row_num = max(cell["row_nums"]) + 1
        max_column_num = max(cell["column_nums"]) + 1
        for row_num in cell["row_nums"]:
            for column_num in cell["column_nums"]:
                cell_grid[row_num][column_num] = [
                    min_column_num - column_num,
                    min_row_num - row_num,
                    max_column_num - column_num,
                    max_row_num - row_num,
                ]

    return cell_grid


def _get_spanning_cell_rows_and_columns(spanning_cells, rows, columns):
    matches_by_spanning_cell = []
    all_matches = set()
    for spanning_cell in spanning_cells:
        row_matches = set()
        column_matches = set()
        for row_num, row in enumerate(rows):
            bbox1 = [
                spanning_cell["bbox"][0],
                row["bbox"][1],
                spanning_cell["bbox"][2],
                row["bbox"][3],
            ]
            bbox2 = _Rect(spanning_cell["bbox"]).intersect(bbox1)
            if bbox2.get_area() / _Rect(bbox1).get_area() >= 0.5:
                row_matches.add(row_num)
        for column_num, column in enumerate(columns):
            bbox1 = [
                column["bbox"][0],
                spanning_cell["bbox"][1],
                column["bbox"][2],
                spanning_cell["bbox"][3],
            ]
            bbox2 = _Rect(spanning_cell["bbox"]).intersect(bbox1)
            if bbox2.get_area() / _Rect(bbox1).get_area() >= 0.5:
                column_matches.add(column_num)
        already_taken = False
        this_matches = []
        for row_num in row_matches:
            for column_num in column_matches:
                this_matches.append((row_num, column_num))
                if (row_num, column_num) in all_matches:
                    already_taken = True
        if not already_taken:
            for match in this_matches:
                all_matches.add(match)
            matches_by_spanning_cell.append(this_matches)
            row_nums = [elem[0] for elem in this_matches]
            column_nums = [elem[1] for elem in this_matches]
            row_rect = _Rect()
            for row_num in row_nums:
                row_rect = row_rect.include_rect(rows[row_num]["bbox"])
            column_rect = _Rect()
            for column_num in column_nums:
                column_rect = column_rect.include_rect(columns[column_num]["bbox"])
            intersected_row_column_rect = row_rect.intersect(column_rect)
            spanning_cell["bbox"] = [
                intersected_row_column_rect.x0,
                intersected_row_column_rect.y0,
                intersected_row_column_rect.x1,
                intersected_row_column_rect.y1,
            ]
        else:
            matches_by_spanning_cell.append([])

    return matches_by_spanning_cell


def _output_to_dilatedbbox_grid(bboxes, labels):
    rows = [{"bbox": bbox} for bbox, label in zip(bboxes, labels) if label == 2]
    columns = [{"bbox": bbox} for bbox, label in zip(bboxes, labels) if label == 1]
    spanning_cells = [
        {"bbox": bbox, "score": 1}
        for bbox, label in zip(bboxes, labels)
        if label in [4, 5]
    ]
    rows.sort(key=lambda x: x["bbox"][1] + x["bbox"][3])
    columns.sort(key=lambda x: x["bbox"][0] + x["bbox"][2])
    spanning_cells.sort(key=lambda x: -x["score"])
    cell_grid = []
    for _, row in enumerate(rows):
        column_grid = []
        for _, column in enumerate(columns):
            bbox = _Rect(row["bbox"]).intersect(column["bbox"])
            column_grid.append([bbox.x0, bbox.y0, bbox.x1, bbox.y1])
        cell_grid.append(column_grid)
    matches_by_spanning_cell = _get_spanning_cell_rows_and_columns(
        spanning_cells, rows, columns
    )
    for matches, spanning_cell in zip(matches_by_spanning_cell, spanning_cells):
        for match in matches:
            cell_grid[match[0]][match[1]] = spanning_cell["bbox"]

    return cell_grid


def grits_top_score(
    y_true: str,
    y_pred: str,
    ignored_nodes: list[str] = None,
    return_components: bool = False,
) -> float | tuple[float, float, float]:
    """Compute GriTS_Top (topology) score for two HTML tables.
    Measures how well the table structure is preserved.

    Args:
        y_true (str): Ground truth HTML table
        y_pred (str): Predicted HTML table
        ignored_nodes (list[str], optional): List of HTML tags to remove. Defaults to None.
        return_components (bool, optional): If True, return fscore, precision, recall. If False, return only the F-score. Defaults to False.

    Returns:
        float | tuple[float, float, float]: F-score (float) or tuple (fscore, precision, recall)

    Example:
        >>> true_html = "<table><tr><td colspan='2'>A</td></tr></table>"
        >>> pred_html = "<table><tr><td>A</td><td>B</td></tr></table>"
        >>> score = grits_top_score(true_html, pred_html)
        >>> print(f"Topology F-score: {score:.3f}")
    """
    true_cells = html_to_cells(y_true, ignored_nodes=ignored_nodes)
    pred_cells = html_to_cells(y_pred, ignored_nodes=ignored_nodes)

    if not true_cells or not pred_cells:
        if return_components:
            return 0.0, 0.0, 0.0
        return 0.0

    true_grid = np.array(_cells_to_relspan_grid(true_cells))
    pred_grid = np.array(_cells_to_relspan_grid(pred_cells))

    fscore, precision, recall, _ = _factored_2dmss(
        true_grid, pred_grid, reward_function=_iou
    )

    if return_components:
        return fscore, precision, recall
    else:
        return fscore


def grits_loc_score(
    y_true_bboxes: list[list[float]],
    y_true_labels: list[int],
    y_pred_bboxes: list[list[float]],
    y_pred_labels: list[int],
    return_components: bool = False,
) -> float | tuple[float, float, float]:
    """Compute GriTS_Loc (location) score using bounding boxes and structural labels.

    GriTS_Loc measures how accurately the spatial layout of table cells
    is preserved between ground truth and prediction. The metric aligns
    table rows and columns using dynamic programming and compares the
    resulting cell bounding boxes using Intersection-over-Union (IoU).

    This implementation follows the PubTables-1M / GriTS definition and
    internally reconstructs the table grid from raw bounding boxes and labels.

    Label convention (as used in PubTables-1M):
        - 1: column
        - 2: row
        - 4: spanning cell
        - 5: spanning cell (alternative type)

    Args:
        y_true_bboxes (list[list[float]]): Ground truth bounding boxes in [x0, y0, x1, y1] format
        y_true_labels (list[int]): Ground truth labels corresponding to each bounding box
        y_pred_bboxes (list[list[float]]): Predicted bounding boxes in [x0, y0, x1, y1] format
        y_pred_labels (list[int]): Predicted labels corresponding to each bounding box
        return_components (bool, optional): If True, return fscore, precision, recall. If False, return only the F-score. Defaults to False.

    Returns:
        float | tuple[float, float, float]: F-score (float) or tuple (fscore, precision, recall)

    Example:
        >>> y_true_bboxes = [
        ...     [0, 0, 100, 20],
        ...     [0, 20, 100, 40],
        ...     [0, 0, 50, 40],
        ...     [50, 0, 100, 40],
        ... ]
        >>> y_true_labels = [1, 1, 2, 2] # column, column, row, row
        >>> y_pred_bboxes = [
        ...     [0, 0, 98, 22],
        ...     [0, 22, 98, 42],
        ...     [2, 0, 52, 40],
        ...     [52, 0, 102, 40],
        ... ]
        >>> y_pred_labels = [1, 1, 2, 2]
        >>> score = grits_loc_score(
        ...     y_true_bboxes,
        ...     y_true_labels,
        ...     y_pred_bboxes,
        ...     y_pred_labels,
        ... )
        >>> print(f"GriTS_Loc F-score: {score:.3f}")
    """
    if not y_true_bboxes or not y_true_labels or not y_pred_bboxes or not y_pred_labels:
        if return_components:
            return 0.0, 0.0, 0.0
        return 0.0

    true_grid = np.array(
        _output_to_dilatedbbox_grid(y_true_bboxes, y_true_labels), dtype=object
    )
    pred_grid = np.array(
        _output_to_dilatedbbox_grid(y_pred_bboxes, y_pred_labels), dtype=object
    )

    fscore, precision, recall, _ = _factored_2dmss(
        true_grid, pred_grid, reward_function=_iou
    )

    if return_components:
        return fscore, precision, recall
    else:
        return fscore


def grits_con_score(
    y_true: str,
    y_pred: str,
    ignored_nodes: list[str] = None,
    return_components: bool = False,
) -> float | tuple[float, float, float]:
    """Compute GriTS_Con (content) score for two HTML tables.
    Measures text content similarity between cells.

    Args:
        y_true (str): Ground truth HTML table
        y_pred (str): Predicted HTML table
        ignored_nodes (list[str], optional): List of HTML tags to remove. Defaults to None.
        return_components (bool, optional): If True, return fscore, precision, recall. If False, return only the F-score. Defaults to False.

    Returns:
        float | tuple[float, float, float]: F-score (float) or tuple (fscore, precision, recall)

    Example:
        >>> true_html = "<table><tr><td>A</td><td>B</td></tr></table>"
        >>> pred_html = "<table><tr><td>A</td><td>B</td></tr></table>"
        >>> score = grits_con_score(true_html, pred_html)
        >>> print(f"Content F-score: {score:.3f}")
    """
    true_cells = html_to_cells(y_true, ignored_nodes=ignored_nodes)
    pred_cells = html_to_cells(y_pred, ignored_nodes=ignored_nodes)

    if not true_cells or not pred_cells:
        if return_components:
            return 0.0, 0.0, 0.0
        return 0.0

    true_grid = np.array(_cells_to_grid(true_cells, key="cell_text"), dtype=object)
    pred_grid = np.array(_cells_to_grid(pred_cells, key="cell_text"), dtype=object)

    fscore, precision, recall, _ = _factored_2dmss(
        true_grid, pred_grid, reward_function=_lcs_similarity
    )

    if return_components:
        return fscore, precision, recall
    else:
        return fscore


def html_to_cells(
    table_html: str, ignored_nodes: list[str] = None
) -> list[dict[str, int | bool | str]]:
    """Parse an HTML table into a flat list of logical table cells.

    This function converts an HTML table into a list of cell descriptors,
    where each cell is mapped to its corresponding row and column indices
    in the table grid. The output format is compatible with GriTS_Top and
    GriTS_Con metrics.

    Args:
        table_html (str): HTML string containing a table. Only the first <table> element is parsed
        ignored_nodes (list[str], optional): List of HTML tag names to strip from the table before parsing. Defaults to None.

    Returns:
        list[dict[str, int | bool | str]]: A list of cell dictionaries.
        Each dictionary has the following keys:
        - "row_nums" (list[int]):
            List of row indices occupied by the cell.
        - "column_nums" (list[int]):
            List of column indices occupied by the cell.
        - "is_column_header" (bool):
            True if the cell is a column header (i.e. <th> element or located inside <thead>).
        - "cell_text" (str):
            Text content of the cell, obtained by concatenating all descendant text nodes.

        Row and column indices are zero-based.

    Example:
        >>> html_table = '''
        ... <table>
        ...   <tr>
        ...     <th>A</th>
        ...     <th>B</th>
        ...   </tr>
        ...   <tr>
        ...     <td colspan="2">C</td>
        ...   </tr>
        ... </table>
        ... '''
        >>> cells = html_to_cells(html_table)
        >>> cells
        [
            {
                "row_nums": [0],
                "column_nums": [0],
                "is_column_header": True,
                "cell_text": "A"
            },
            {
                "row_nums": [0],
                "column_nums": [1],
                "is_column_header": True,
                "cell_text": "B"
            },
            {
                "row_nums": [1],
                "column_nums": [0, 1],
                "is_column_header": False,
                "cell_text": "C"
            }
        ]
    """
    html_parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    xpath_element = "//table[1]"

    tree = html.fromstring(table_html, parser=html_parser)
    table = tree.xpath(xpath_element)
    if not table:
        return []

    table = table[0]
    if ignored_nodes:
        etree.strip_tags(table, *ignored_nodes)

    table_cells = []

    occupied_columns_by_row = defaultdict(set)
    current_row = -1

    stack = [(table, False)]
    while stack:
        current, in_header = stack.pop()
        current_tag = current.tag

        if current_tag == "tr":
            current_row += 1
        elif current_tag in ("td", "th"):
            colspan = int(current.attrib.get("colspan", 1))
            rowspan = int(current.attrib.get("rowspan", 1))

            occupied_columns = occupied_columns_by_row[current_row]
            current_column = 0
            if occupied_columns:
                while current_column in occupied_columns:
                    current_column += 1

            row_nums = list(range(current_row, current_row + rowspan))
            column_nums = list(range(current_column, current_column + colspan))
            for row_num in row_nums:
                occupied_columns_by_row[row_num].update(column_nums)

            cell_dict = {
                "row_nums": row_nums,
                "column_nums": column_nums,
                "is_column_header": current_tag == "th" or in_header,
                "cell_text": " ".join(current.itertext()),
            }

            table_cells.append(cell_dict)

        children = list(current)
        for child in reversed(children):
            stack.append((child, in_header or current_tag in ("th", "thead")))

    return table_cells
