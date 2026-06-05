"""
Output writing tool for persisting agent results to files.

What it does:
    Provides a WriteOutputTool that writes structured data to files in JSON,
    CSV, or append-JSON formats. Handles file creation, directory setup,
    and proper serialization.

Entities in it:
    - WriteOutputTool: Concrete tool for writing output data to files.

How used by other modules:
    - Registered in the ToolRegistry at application startup.
    - Called by agents during workflow execution to persist results.
    - Used at the end of workflow pipelines to store final outputs.
"""

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from backend.tools.base import BaseTool, ToolExecutionError


class WriteOutputTool(BaseTool):
    """
    Writes structured data to files in JSON, CSV, or append-JSON formats.

    Description:
        Supports three output formats: 'json' for complete JSON file writes,
        'csv' for tabular data, and 'append_json' for appending records to
        an existing JSON array file.

    Attributes:
        SUPPORTED_FORMATS: Class-level set of supported output formats.

    Methods:
        name: Returns 'write_output'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for output writing parameters.
        execute: Writes data to the specified file in the specified format.
    """

    SUPPORTED_FORMATS = {"json", "csv", "append_json"}

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'write_output'
        """
        return "write_output"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the output writing formats available.

        Params:
            None

        Returns:
            str: Description string.
        """
        return "Writes structured data to files. Formats: json, csv, append_json."

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines file_path, format, and data parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Output file path"},
                "format": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_FORMATS),
                    "description": "Output format (json, csv, append_json)",
                },
                "data": {
                    "description": "Data to write (object/array for json, array of objects for csv)",
                },
            },
            "required": ["file_path", "format", "data"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Write data to a file in the specified format.

        Description:
            Dispatches to the appropriate writer based on format parameter.

        Params:
            **kwargs (Any): Must include 'file_path', 'format', and 'data'.

        Returns:
            dict: Dictionary with 'written' status, 'file_path', and 'bytes_written'.

        Raises:
            ToolExecutionError: If format is unsupported or I/O errors occur.
        """
        file_path_str = kwargs.get("file_path")
        output_format = kwargs.get("format")
        data = kwargs.get("data")

        if not file_path_str:
            raise ToolExecutionError("file_path is required")
        if output_format not in self.SUPPORTED_FORMATS:
            raise ToolExecutionError(
                f"Unsupported format: '{output_format}'. "
                f"Must be one of: {sorted(self.SUPPORTED_FORMATS)}"
            )
        if data is None:
            raise ToolExecutionError("data is required")

        file_path = Path(file_path_str)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if output_format == "json":
                return self._write_json(file_path, data)
            elif output_format == "csv":
                return self._write_csv(file_path, data)
            elif output_format == "append_json":
                return self._write_append_json(file_path, data)
        except OSError as io_error:
            raise ToolExecutionError(
                f"I/O error writing to {file_path}: {io_error}"
            ) from io_error

        raise ToolExecutionError(f"Unhandled format: '{output_format}'")

    def _write_json(self, file_path: Path, data: Any) -> dict:
        """
        Write data as a formatted JSON file.

        Description:
            Serializes data to JSON with indentation and writes to file.

        Params:
            file_path (Path): Target file path.
            data (Any): Data to serialize.

        Returns:
            dict: Write result with status and byte count.
        """
        content = json.dumps(data, indent=2, default=str)
        file_path.write_text(content, encoding="utf-8")
        return {
            "written": True,
            "file_path": str(file_path),
            "bytes_written": len(content.encode("utf-8")),
        }

    def _write_csv(self, file_path: Path, data: Any) -> dict:
        """
        Write data as a CSV file.

        Description:
            Expects data as a list of dictionaries. Uses the keys from the
            first record as column headers.

        Params:
            file_path (Path): Target file path.
            data (Any): List of dictionaries to write as CSV rows.

        Returns:
            dict: Write result with status and byte count.

        Raises:
            ToolExecutionError: If data is not a list of dictionaries.
        """
        if not isinstance(data, list) or not data:
            raise ToolExecutionError("CSV format requires data to be a non-empty list of objects")
        if not isinstance(data[0], dict):
            raise ToolExecutionError("CSV format requires each item in data to be a dictionary")

        output_buffer = StringIO()
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(output_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

        content = output_buffer.getvalue()
        file_path.write_text(content, encoding="utf-8")
        return {
            "written": True,
            "file_path": str(file_path),
            "bytes_written": len(content.encode("utf-8")),
        }

    def _write_append_json(self, file_path: Path, data: Any) -> dict:
        """
        Append data to an existing JSON array file.

        Description:
            Reads the existing JSON array from the file (or starts with an
            empty array if the file doesn't exist), appends the new data,
            and writes the updated array back.

        Params:
            file_path (Path): Target file path.
            data (Any): Data to append (single item or list of items).

        Returns:
            dict: Write result with status and byte count.
        """
        existing_data: list = []
        if file_path.exists():
            existing_content = file_path.read_text(encoding="utf-8")
            if existing_content.strip():
                existing_data = json.loads(existing_content)
                if not isinstance(existing_data, list):
                    raise ToolExecutionError(
                        f"append_json expects existing file to contain a JSON array, "
                        f"but found {type(existing_data).__name__}"
                    )

        if isinstance(data, list):
            existing_data.extend(data)
        else:
            existing_data.append(data)

        content = json.dumps(existing_data, indent=2, default=str)
        file_path.write_text(content, encoding="utf-8")
        return {
            "written": True,
            "file_path": str(file_path),
            "bytes_written": len(content.encode("utf-8")),
        }
