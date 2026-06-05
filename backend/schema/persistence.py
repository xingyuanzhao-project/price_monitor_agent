"""
YAML-based persistence for workflow schemas.

What it does:
    Provides file-system persistence for WorkflowSchema objects using YAML
    serialization. Supports save, load, list, and delete operations against
    a configured schema directory.

Entities in it:
    - SchemaPersistence: Main class handling all schema file I/O operations.

How used by other modules:
    - The frontend calls save_schema/load_schema when users create or open workflows.
    - The orchestration engine calls load_schema to retrieve a schema for execution.
    - The schema management UI calls list_schemas/delete_schema for browsing.
"""

from pathlib import Path

import yaml

from backend.schema.models import WorkflowSchema


class SchemaPersistence:
    """
    Manages reading and writing WorkflowSchema objects to YAML files.

    Description:
        Provides CRUD-like operations for workflow schemas stored as YAML files
        in a designated directory. Each schema is stored in a file named by its
        schema_id with a .yaml extension.

    Attributes:
        schema_directory: Path to the directory where schema YAML files are stored.

    Methods:
        save_schema: Serialize and write a WorkflowSchema to a YAML file.
        load_schema: Read and deserialize a WorkflowSchema from a YAML file.
        list_schemas: Return identifiers of all stored schemas.
        delete_schema: Remove a schema YAML file from disk.
    """

    def __init__(self, schema_directory: Path) -> None:
        """
        Initialize SchemaPersistence with the target directory.

        Description:
            Sets the directory path and ensures it exists on the filesystem.

        Params:
            schema_directory (Path): Directory where schema YAML files are stored.

        Returns:
            None
        """
        self.schema_directory = schema_directory
        self.schema_directory.mkdir(parents=True, exist_ok=True)

    def _schema_file_path(self, schema_id: str) -> Path:
        """
        Compute the file path for a given schema identifier.

        Description:
            Constructs the full path by joining the schema directory with the
            schema_id and .yaml extension.

        Params:
            schema_id (str): Unique identifier of the schema.

        Returns:
            Path: Full filesystem path to the schema YAML file.
        """
        return self.schema_directory / f"{schema_id}.yaml"

    def save_schema(self, schema: WorkflowSchema) -> Path:
        """
        Serialize and persist a WorkflowSchema to a YAML file.

        Description:
            Converts the schema to a dictionary and writes it as YAML to the
            appropriate file in the schema directory.

        Params:
            schema (WorkflowSchema): The workflow schema object to persist.

        Returns:
            Path: The file path where the schema was saved.
        """
        file_path = self._schema_file_path(schema.schema_id)
        schema_data = schema.model_dump(mode="json")
        with open(file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(schema_data, yaml_file, default_flow_style=False, sort_keys=False)
        return file_path

    def load_schema(self, schema_id: str) -> WorkflowSchema:
        """
        Load and deserialize a WorkflowSchema from a YAML file.

        Description:
            Reads the YAML file for the given schema_id and constructs a
            WorkflowSchema instance from the data.

        Params:
            schema_id (str): Unique identifier of the schema to load.

        Returns:
            WorkflowSchema: The deserialized workflow schema object.

        Raises:
            FileNotFoundError: If no schema file exists for the given schema_id.
        """
        file_path = self._schema_file_path(schema_id)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {file_path}. "
                f"No schema exists with id '{schema_id}' in directory '{self.schema_directory}'."
            )
        with open(file_path, "r", encoding="utf-8") as yaml_file:
            schema_data = yaml.safe_load(yaml_file)
        return WorkflowSchema.model_validate(schema_data)

    def list_schemas(self) -> list[str]:
        """
        List identifiers of all schemas stored in the schema directory.

        Description:
            Scans the schema directory for .yaml files and extracts schema_ids
            from their filenames.

        Params:
            None

        Returns:
            list[str]: List of schema_id strings for all stored schemas.
        """
        schema_files = self.schema_directory.glob("*.yaml")
        return [file_path.stem for file_path in sorted(schema_files)]

    def delete_schema(self, schema_id: str) -> None:
        """
        Remove a schema YAML file from the schema directory.

        Description:
            Deletes the file corresponding to the given schema_id.

        Params:
            schema_id (str): Unique identifier of the schema to delete.

        Returns:
            None

        Raises:
            FileNotFoundError: If no schema file exists for the given schema_id.
        """
        file_path = self._schema_file_path(schema_id)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {file_path}. "
                f"Cannot delete schema with id '{schema_id}' — it does not exist "
                f"in directory '{self.schema_directory}'."
            )
        file_path.unlink()
