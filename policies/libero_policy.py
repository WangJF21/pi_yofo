import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Yofo policy."""
    return {
        "observation/state": np.random.rand(7),  # Yofo: 6 joints + 1 gripper = 7D
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Yofo robot has only one wrist camera, no base camera.
        # We use the wrist image for both base and left_wrist views to maximize
        # the model's ability to understand the scene from the available camera.
        wrist_image = _parse_image(data["observation/wrist_image"])

        # For Yofo: use wrist camera image for both base and left_wrist inputs
        # This allows the model to process the single camera view through multiple pathways
        inputs = {
            "state": data["observation/state"],  # 7D: 6 joints + 1 gripper
            "image": {
                "base_0_rgb": wrist_image,  # Yofo: reuse wrist camera
                "left_wrist_0_rgb": wrist_image,  # Yofo: actual wrist camera
                # Pad right wrist with zeros since Yofo doesn't have it
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,  # Enable: using wrist camera
                "left_wrist_0_rgb": np.True_,  # Enable: actual wrist camera
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Yofo robot: 6 joint angles + 1 gripper = 7 actions total.
        # This matches the robot's action space: [joint1, joint2, ..., joint6, gripper]
        return {"actions": np.asarray(data["actions"][:, :7])}
