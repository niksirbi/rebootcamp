import copy
from collections import UserDict
from pathlib import Path
from typing import Any, Literal, Union

import torch
import yaml

# Define type aliases
MyPath = Union[str, Path]


class DiffusionConfig(UserDict):

    dtype: torch.dtype
    model_id: str
    scheduler_id: Literal["euler", "linear"]
    output_dir: MyPath
    test_prefix: str
    negative_prompt_path: MyPath

    def __init__(self, cfg_path, input_dict):
        super(DiffusionConfig, self).__init__(input_dict)

        self.cfg_path = cfg_path

    def dump_to_file(self):
        """
        Dump the config to a yaml file.
        """
        cfg_to_save = copy.deepcopy(self.data)
        self.convert_paths(cfg_to_save, "path2str")

        with open(self.file_path, "w") as cfg_file:
            yaml.dump(cfg_to_save, cfg_file, sort_keys=False)

    def load_from_file(self):
        """
        Load the config from a yaml file.
        """
        with open(self.file_path, "r") as cfg_file:
            cfg_dict = yaml.full_load(cfg_file)

        self.convert_paths(cfg_dict, "str2path")

        self.data = cfg_dict

    def update_an_entry(self, option_key: str, new_info: Any):
        """
        Convenience function to update individual entry of configuration
        file. The config file, and currently loaded self.cfg will be
        updated.
        In case an update is breaking, revert to the original value.

        Parameters
        ----------
        option_key : str
            The key of the option to be updated.
        new_info : Any
        """
        if option_key not in self:
            raise ValueError(f"'{option_key}' is not a valid config.")

        original_value = copy.deepcopy(self[option_key])

        path_keys = [key for key in self if "path" in key]
        if option_key in path_keys:
            new_info = Path(new_info)

        try:
            self[option_key] = new_info
            self.dump_to_file()
        except Warning:
            print(
                f"""
                Update of '{option_key}' failed.
                Reverting to original value {original_value}.
                """
            )

    @staticmethod
    def convert_paths(
        cfg_dict: dict, direction: Literal["str2path", "path2str"]
    ):
        """
        Convert paths to strings and vice versa.

        Parameters
        ----------
        cfg_dict : dict
            The config dictionary.
        direction : str
            The direction of the conversion.
            Must be either "str2path" or "path2str".
        """

        path_keys = [key for key in cfg_dict.keys() if "path" in key]
        for path_key in path_keys:
            value = cfg_dict[path_key]

            if value:
                if direction == "str2path":
                    cfg_dict[path_key] = Path(value)

                elif direction == "path2str":
                    if type(value) != str:
                        cfg_dict[path_key] = value.as_posix()

                else:
                    raise ValueError(
                        "direction must be either 'str2path' or 'path2str'."
                    )
