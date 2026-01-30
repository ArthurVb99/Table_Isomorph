"""
Data Transformation Module for Table/Data Permutations
Handles row-column swapping and permutations on JSON/HTML data.
"""

import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional


class DataTransformer:
    """
    Transform data structures by applying row-column permutations.
    Supports row swaps, column swaps, and combined operations.
    """

    def __init__(self, data: Dict[str, Any] | List[Dict[str, Any]]):
        """
        Initialize the transformer with data.

        Args:
            data: Dictionary, list of dictionaries, or JSON-serializable data
        """
        self.original_data = data
        self.df = self._convert_to_dataframe(data)

    def _convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Convert data to pandas DataFrame.

        Args:
            data: Input data (dict, list of dicts, or DataFrame)

        Returns:
            pd.DataFrame: Converted dataframe
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def swap_rows(self, row1_idx: int, row2_idx: int) -> "DataTransformer":
        """
        Swap two rows by index.

        Args:
            row1_idx (int): Index of first row
            row2_idx (int): Index of second row

        Returns:
            DataTransformer: Self for chaining
        """
        self.df.iloc[[row1_idx, row2_idx]] = self.df.iloc[[row2_idx, row1_idx]].values
        return self

    def swap_columns(self, col1_name: str, col2_name: str) -> "DataTransformer":
        """
        Swap two columns by name.

        Args:
            col1_name (str): Name of first column
            col2_name (str): Name of second column

        Returns:
            DataTransformer: Self for chaining
        """
        self.df[[col1_name, col2_name]] = self.df[[col2_name, col1_name]]
        return self

    def swap_rows_by_position(self, row1: int, row2: int) -> "DataTransformer":
        """
        Swap rows by position (1-indexed for user convenience).

        Args:
            row1 (int): Position of first row (1-indexed)
            row2 (int): Position of second row (1-indexed)

        Returns:
            DataTransformer: Self for chaining
        """
        return self.swap_rows(row1 - 1, row2 - 1)

    def swap_columns_by_position(self, col1: int, col2: int) -> "DataTransformer":
        """
        Swap columns by position (1-indexed for user convenience).

        Args:
            col1 (int): Position of first column (1-indexed)
            col2 (int): Position of second column (1-indexed)

        Returns:
            DataTransformer: Self for chaining
        """
        cols = list(self.df.columns)
        col1_name = cols[col1 - 1]
        col2_name = cols[col2 - 1]
        return self.swap_columns(col1_name, col2_name)

    def apply_row_column_permutations(self, permutations: List[Dict[str, Any]]) -> "DataTransformer":
        """
        Apply multiple row-column permutations from a list of instructions.

        Args:
            permutations (List[Dict]): List of permutation instructions
                Each item can have:
                - 'type': 'row' or 'column' or 'both'
                - 'first': index/name of first element
                - 'second': index/name of second element

        Returns:
            DataTransformer: Self for chaining

        Example:
            ```
            permutations = [
                {'type': 'row', 'first': 1, 'second': 2},
                {'type': 'column', 'first': 'name', 'second': 'age'},
                {'type': 'both', 'rows': [(0, 1), (2, 3)], 'columns': [('col_a', 'col_b')]}
            ]
            transformer.apply_row_column_permutations(permutations)
            ```
        """
        for perm in permutations:
            perm_type = perm.get('type', 'row').lower()

            if perm_type == 'row':
                self.swap_rows_by_position(perm['first'], perm['second'])

            elif perm_type == 'column':
                if isinstance(perm['first'], int):
                    self.swap_columns_by_position(perm['first'], perm['second'])
                else:
                    self.swap_columns(perm['first'], perm['second'])

            elif perm_type == 'both':
                # Apply row swaps
                for row_pair in perm.get('rows', []):
                    self.swap_rows(row_pair[0], row_pair[1])
                # Apply column swaps
                for col_pair in perm.get('columns', []):
                    if isinstance(col_pair[0], int):
                        self.swap_columns_by_position(col_pair[0], col_pair[1])
                    else:
                        self.swap_columns(col_pair[0], col_pair[1])

        return self

    def rotate_rows(self, steps: int = 1) -> "DataTransformer":
        """
        Rotate rows by specified steps.

        Args:
            steps (int): Number of steps to rotate (positive = down, negative = up)

        Returns:
            DataTransformer: Self for chaining
        """
        self.df = self.df.iloc[(-steps) % len(self.df):].reset_index(drop=True)
        self.df = pd.concat([self.df, self.df.iloc[:(steps) % len(self.df)]], ignore_index=True)
        return self

    def rotate_columns(self, steps: int = 1) -> "DataTransformer":
        """
        Rotate columns by specified steps.

        Args:
            steps (int): Number of steps to rotate

        Returns:
            DataTransformer: Self for chaining
        """
        cols = list(self.df.columns)
        rotated_cols = cols[(-steps) % len(cols):] + cols[:((-steps) % len(cols))]
        self.df = self.df[rotated_cols]
        return self

    def shuffle_rows(self, seed: Optional[int] = None) -> "DataTransformer":
        """
        Randomly shuffle rows.

        Args:
            seed (int, optional): Random seed for reproducibility

        Returns:
            DataTransformer: Self for chaining
        """
        self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return self

    def shuffle_columns(self, seed: Optional[int] = None) -> "DataTransformer":
        """
        Randomly shuffle columns.

        Args:
            seed (int, optional): Random seed for reproducibility

        Returns:
            DataTransformer: Self for chaining
        """
        cols = list(self.df.columns)
        import random
        if seed is not None:
            random.seed(seed)
        shuffled_cols = cols.copy()
        random.shuffle(shuffled_cols)
        self.df = self.df[shuffled_cols]
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert transformed data back to dictionary.

        Returns:
            Dict: Dictionary representation
        """
        return self.df.to_dict('records')

    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.

        Args:
            indent (int): JSON indentation

        Returns:
            str: JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get the current DataFrame.

        Returns:
            pd.DataFrame: Current dataframe
        """
        return self.df.copy()

    def get_dataframe(self) -> pd.DataFrame:
        """Alias for to_dataframe()."""
        return self.to_dataframe()

    def display(self) -> None:
        """Display the current data as a formatted table."""
        print(self.df.to_string())


if __name__ == "__main__":
    # Example usage
    data = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
        {"id": 3, "name": "Charlie", "age": 35},
    ]

    transformer = DataTransformer(data)
    print("Original:")
    transformer.display()

    # Swap rows
    print("\nAfter swapping rows 1 and 2:")
    transformer.swap_rows_by_position(1, 2)
    transformer.display()

    # Swap columns
    print("\nAfter swapping columns 'name' and 'age':")
    transformer.swap_columns("name", "age")
    transformer.display()

    # To JSON
    print("\nAs JSON:")
    print(transformer.to_json())
