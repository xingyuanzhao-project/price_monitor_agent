"""
JSON-based persistence for user settings.

What it does:
    Provides file-system persistence for UserSettings objects using JSON
    serialization. Supports save, load, and existence-check operations
    against a configured settings file path.

Entities in it:
    - SettingsPersistence: Main class handling settings file I/O operations.

How used by other modules:
    - The frontend calls save_settings when users modify their configuration.
    - The orchestration engine calls load_settings at startup to retrieve
      credentials and provider configs needed for workflow execution.
    - The initialization flow calls exists() to determine if first-run setup
      is needed.
"""

import json
from pathlib import Path

from backend.settings.models import UserSettings


class SettingsPersistence:
    """
    Manages reading and writing UserSettings objects to a JSON file.

    Description:
        Provides operations to persist and retrieve user settings from a
        single JSON file on the filesystem. Ensures the parent directory
        exists before writing.

    Attributes:
        settings_file_path: Path to the JSON file where settings are stored.

    Methods:
        save_settings: Serialize and write UserSettings to the JSON file.
        load_settings: Read and deserialize UserSettings from the JSON file.
        exists: Check whether the settings file exists on disk.
    """

    def __init__(self, settings_file_path: Path) -> None:
        """
        Initialize SettingsPersistence with the target file path.

        Description:
            Stores the file path for subsequent I/O operations.

        Params:
            settings_file_path (Path): Path to the JSON settings file.

        Returns:
            None
        """
        self.settings_file_path = settings_file_path

    def save_settings(self, settings: UserSettings) -> Path:
        """
        Serialize and persist UserSettings to the JSON file.

        Description:
            Converts the settings to a dictionary and writes it as formatted
            JSON to the configured file path, creating parent directories
            if necessary.

        Params:
            settings (UserSettings): The user settings object to persist.

        Returns:
            Path: The file path where settings were saved.
        """
        self.settings_file_path.parent.mkdir(parents=True, exist_ok=True)
        settings_data = settings.model_dump(mode="json")
        with open(self.settings_file_path, "w", encoding="utf-8") as json_file:
            json.dump(settings_data, json_file, indent=2)
        return self.settings_file_path

    def load_settings(self) -> UserSettings:
        """
        Load and deserialize UserSettings from the JSON file.

        Description:
            Reads the JSON file and constructs a UserSettings instance
            from the data.

        Params:
            None

        Returns:
            UserSettings: The deserialized user settings object.

        Raises:
            FileNotFoundError: If the settings file does not exist.
        """
        if not self.settings_file_path.exists():
            raise FileNotFoundError(
                f"Settings file not found: {self.settings_file_path}. "
                f"No settings have been saved yet."
            )
        with open(self.settings_file_path, "r", encoding="utf-8") as json_file:
            settings_data = json.load(json_file)
        return UserSettings.model_validate(settings_data)

    def exists(self) -> bool:
        """
        Check whether the settings file exists on disk.

        Description:
            Returns True if the settings JSON file is present at the
            configured path, False otherwise.

        Params:
            None

        Returns:
            bool: True if the settings file exists, False otherwise.
        """
        return self.settings_file_path.exists()
