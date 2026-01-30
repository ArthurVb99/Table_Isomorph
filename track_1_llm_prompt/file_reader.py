"""
File Reader Module for JSON and HTML Data
Extracts structured data from JSON and HTML files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Union
from bs4 import BeautifulSoup
import pandas as pd


class DataFileReader:
    """
    Read and parse JSON and HTML files into structured data.
    """

    @staticmethod
    def read_json(file_path: str) -> Dict[str, Any] | List[Dict[str, Any]]:
        """
        Read and parse JSON file.

        Args:
            file_path (str): Path to JSON file

        Returns:
            Dict or List: Parsed JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {str(e)}")

    @staticmethod
    def read_html(file_path: str, table_index: int = 0) -> List[Dict[str, Any]]:
        """
        Read HTML file and extract table data.

        Args:
            file_path (str): Path to HTML file
            table_index (int): Index of table to extract (0 = first table)

        Returns:
            List[Dict]: List of rows as dictionaries

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If no table found or invalid HTML
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table')

            if not tables:
                raise ValueError("No tables found in HTML file")

            if table_index >= len(tables):
                raise ValueError(f"Table index {table_index} out of range. Found {len(tables)} tables.")

            table = tables[table_index]
            
            # Extract headers
            headers = []
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
            else:
                # Try first row
                first_row = table.find('tr')
                if first_row:
                    headers = [th.get_text(strip=True) for th in first_row.find_all('th')]
                    if not headers:
                        headers = [td.get_text(strip=True) for td in first_row.find_all('td')]

            # Extract rows
            rows = []
            tbody = table.find('tbody')
            table_rows = tbody.find_all('tr') if tbody else table.find_all('tr')[1:]

            for row in table_rows:
                cells = row.find_all('td')
                if cells:
                    row_data = {}
                    for i, cell in enumerate(cells):
                        header = headers[i] if i < len(headers) else f"col_{i}"
                        row_data[header] = cell.get_text(strip=True)
                    rows.append(row_data)

            return rows

        except Exception as e:
            raise ValueError(f"Error reading HTML file {file_path}: {str(e)}")

    @staticmethod
    def read_html_all_tables(file_path: str) -> List[List[Dict[str, Any]]]:
        """
        Read HTML file and extract all tables.

        Args:
            file_path (str): Path to HTML file

        Returns:
            List[List[Dict]]: List of tables, each containing rows

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')

        all_tables = []
        for table in tables:
            # Extract headers
            headers = []
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
            else:
                first_row = table.find('tr')
                if first_row:
                    headers = [th.get_text(strip=True) for th in first_row.find_all('th')]
                    if not headers:
                        headers = [td.get_text(strip=True) for td in first_row.find_all('td')]

            # Extract rows
            rows = []
            tbody = table.find('tbody')
            table_rows = tbody.find_all('tr') if tbody else table.find_all('tr')[1:]

            for row in table_rows:
                cells = row.find_all('td')
                if cells:
                    row_data = {}
                    for i, cell in enumerate(cells):
                        header = headers[i] if i < len(headers) else f"col_{i}"
                        row_data[header] = cell.get_text(strip=True)
                    rows.append(row_data)

            if rows:
                all_tables.append(rows)

        return all_tables

    @staticmethod
    def read_csv(file_path: str) -> List[Dict[str, Any]]:
        """
        Read CSV file and return as list of dictionaries.

        Args:
            file_path (str): Path to CSV file

        Returns:
            List[Dict]: List of rows as dictionaries
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        return df.to_dict('records')

    @staticmethod
    def read_file(file_path: str, **kwargs) -> Union[Dict, List]:
        """
        Auto-detect file type and read accordingly.

        Args:
            file_path (str): Path to file
            **kwargs: Additional arguments passed to specific readers

        Returns:
            Parsed data from file

        Raises:
            ValueError: If file type not supported
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.json':
            return DataFileReader.read_json(file_path)
        elif file_ext == '.html' or file_ext == '.htm':
            return DataFileReader.read_html(file_path, **kwargs)
        elif file_ext == '.csv':
            return DataFileReader.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")


if __name__ == "__main__":
    # Example usage
    # Read JSON
    # data = DataFileReader.read_json("data.json")
    # print(json.dumps(data, indent=2))

    # Read HTML table
    # table_data = DataFileReader.read_html("table.html")
    # print(json.dumps(table_data, indent=2))

    print("DataFileReader initialized. Use to read JSON/HTML/CSV files.")
