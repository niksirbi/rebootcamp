from pathlib import Path

# import matplotlib.pyplot as plt
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

from rebootcamp.utils import get_device, load_prompts

# Specify paths for outputs
data_dir = Path(__name__).parent / "data"
tmp_dir = data_dir / "tmp"

# Get device (use GPU if available)
device = get_device()

# Set model ID
model_id = "stabilityai/stable-diffusion-2-1"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler"
)

# Choose the appropriate dtype
dtype = torch.float16
if device == torch.device("cpu"):
    dtype = torch.float32

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, revision="fp16", torch_dtype=dtype
)
# Send model to device
pipe = pipe.to(device)

# Reduces memory usage
pipe.enable_attention_slicing()

# Set prompts
positive_prompt = "The quick brown fox jumps over the lazy dog."
negative_prompt = load_prompts(data_dir / "negative_prompts.txt")

# Set height and width
height = 1280
width = 720
# Check inputs. Raise error if not correct
pipe.check_inputs(positive_prompt, height, width, 1)

# Define call parameters
batch_size = 1
device = pipe._execution_device
guidance_scale = 9.5
if guidance_scale > 1:
    do_classifier_free_guidance = True

# Encode input prompt
num_images_per_prompt = 1
text_embeddings = pipe._encode_prompt(
    positive_prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt,
)
