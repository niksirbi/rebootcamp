import copy
from collections import UserDict
from pathlib import Path
from typing import Any, Literal, Union

import yaml

from rebootcamp.utils import get_device

# Define type aliases
MyPath = Union[str, Path]

# Get the default device based on GPU availability
default_device = get_device()

# Get path to the negative prompts file
data_dir = Path(__name__).parent / "data"
data_dir = data_dir.resolve()
negative_prompts_path = data_dir / "negative_prompts.txt"
print(negative_prompts_path)


DEFAULT_DIFF_DICT = {
    "positive_prompt": "hippocampus",
    "dtype": "torch.float16",
    "model_id": "stabilityai/stable-diffusion-2-1",
    "scheduler_id": "EulerDiscreteScheduler",
    "num_inference_steps": 20,
    "seed": 42,
    "output_dir": "/mnt/Data/stable-diffusion",
    "run_prefix": "run_",
    "negative_prompts_path": negative_prompts_path.as_posix(),
    "device": str(default_device),
    "attention_slicing": False,
    "classifier_free_guidance": True,
    "image_height": 512,
    "image_width": 512,
    "batch_size": 1,
    "guidance_scale": 10.0,
    "plot_latents": False,
}


class DiffusionConfig(UserDict):
    """
    A class to handle the configuration of the diffusion model.
    """

    input_path: dict

    def __init__(self, input_dict):
        super(DiffusionConfig, self).__init__(input_dict)

    @classmethod
    def from_file(cls, file_path: MyPath):
        """
        Load the config from a yaml file.

        Parameters
        ----------
        file_path : str or pathlib.Path
            The path to the config file.

        Returns
        -------
        DiffusionConfig
            The configuration.
        """
        with open(file_path, "r") as cfg_file:
            try:
                cfg_dict = yaml.safe_load(cfg_file)
            except Warning:
                """
                Unable to load the config file.
                Make sure that the path leads to a valid yaml file.
                Keeping the default config.
                """
        cls.convert_paths(cfg_dict, "str2path")

        return cls(cfg_dict)

    def dump_to_file(self, file_path: MyPath):
        """
        Dump the config to a yaml file.
        """
        cfg_to_save = copy.deepcopy(self.data)
        self.convert_paths(cfg_to_save, "path2str")

        with open(file_path, "w") as cfg_file:
            yaml.dump(cfg_to_save, cfg_file, sort_keys=False)

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


# create the default config object
DEFAULT_DIFF_CONFIG = DiffusionConfig(DEFAULT_DIFF_DICT)

# Save the default config to a yaml file
# config_path = data_dir / "default_diffusion_config.yaml"
# DEFAULT_DIFF_CONFIG.dump_to_file(config_path)
