from apted import APTED, Config
from apted.helpers import Tree
from rapidfuzz.distance import Levenshtein
from lxml import html, etree
from lxml.html import HtmlElement


class _TableTree(Tree):
    def __init__(
        self,
        tag: str,
        colspan: int | None = None,
        rowspan: int | None = None,
        content: list[str] | None = None,
        *children: tuple[Tree],
    ):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)


class _CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1"""
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1: _TableTree, node2: _TableTree) -> float:
        """Compares attributes of trees"""
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0

        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)

        return 0.0


def teds_score(
    y_true: str,
    y_pred: str,
    structure_only: bool = False,
    ignored_nodes: list[str] = None,
) -> float:
    """Compute TEDS (Tree Edit Distance-based Similarity) between two HTML tables.

    The metric compares the structural and (optionally) textual similarity
    of two HTML table representations using tree edit distance.

    Args:
        y_true (str): Ground-truth HTML string containing a table.
        y_pred (str): Predicted HTML string containing a table.
        structure_only (bool, optional): If True, only the table structure is compared and cell content
        is ignored. Defaults to False.
        ignored_nodes (list[str], optional): List of HTML tags to remove before comparison. Defaults to None.

    Returns:
        float: TEDS similarity score in the range [0, 1], where 1 indicates
        identical tables. Returns 0 if no table is found.
    """
    if not y_true or not y_pred:
        return 0.0

    html_parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
    y_true_element = html.fromstring(y_true, parser=html_parser)
    y_pred_element = html.fromstring(y_pred, parser=html_parser)

    xpath_element = "//table[1]"
    if y_true_element.xpath(xpath_element) and y_pred_element.xpath(xpath_element):
        y_true_element = y_true_element.xpath(xpath_element)[0]
        y_pred_element = y_pred_element.xpath(xpath_element)[0]

        if ignored_nodes:
            etree.strip_tags(y_true_element, *ignored_nodes)
            etree.strip_tags(y_pred_element, *ignored_nodes)

        tree_y_true = _load_html_tree(y_true_element, structure_only=structure_only)
        tree_y_pred = _load_html_tree(y_pred_element, structure_only=structure_only)
        distance = APTED(
            tree_y_true, tree_y_pred, _CustomConfig()
        ).compute_edit_distance()

        n_nodes_true = len(y_true_element.xpath(".//*"))
        n_nodes_pred = len(y_pred_element.xpath(".//*"))
        n_nodes = max(n_nodes_pred, n_nodes_true)

        if n_nodes:
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 1.0

    return 0.0


def _load_html_tree(
    node: HtmlElement,
    parent: _TableTree | None = None,
    html_tokens: list[str] = None,
    structure_only: bool = False,
) -> _TableTree | None:
    if node.tag == "td":
        if structure_only:
            cell = []
        else:
            html_tokens = []
            html_tokens = _tokenize(node, html_tokens)
            cell = html_tokens[1:-1]

        new_node = _TableTree(
            node.tag,
            int(node.attrib.get("colspan", "1")),
            int(node.attrib.get("rowspan", "1")),
            cell,
        )
    else:
        new_node = _TableTree(node.tag, None, None, None)

    if parent is not None:
        parent.children.append(new_node)

    if node.tag != "td":
        for n in node.getchildren():
            html_tokens = _load_html_tree(n, new_node, html_tokens, structure_only)

    if parent is None:
        return new_node

    return None


def _tokenize(
    node: HtmlElement,
    html_tokens: list[str] = None,
) -> list[str]:
    html_tokens.append(f"<{node.tag}>")

    if node.text is not None:
        html_tokens += list(node.text)

    for n in node.getchildren():
        html_tokens = _tokenize(n, html_tokens)

    if node.tag != "unk":
        html_tokens.append(f"</{node.tag}>")

    if node.tag != "td" and node.tail is not None:
        html_tokens += list(node.tail)

    return html_tokens
