from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

from rebootcamp.utils import get_device, load_prompts, plot_latents

# Specify paths for outputs
data_dir = Path(__name__).parent / "data"
tmp_dir = data_dir / "_tmp"

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
height = 960
width = 720
# Check inputs. Raise error if not correct
pipe.check_inputs(positive_prompt, height, width, 1)

# Define call parameters
batch_size = 1
device = pipe._execution_device
guidance_scale = 9.5
if guidance_scale > 1:
    do_classifier_free_guidance = True

# Encode input text prompt
num_images_per_prompt = 1
text_embeddings = pipe._encode_prompt(
    positive_prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt,
)

print(text_embeddings.shape)

# Prepare timesteps
num_inference_steps = 50
pipe.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = pipe.scheduler.timesteps

# Specify random generator seed
seed = 42
generator = torch.Generator(device).manual_seed(seed)

# Prepare latent variables (random initialize)
latents = None
num_channels_latents = pipe.unet.in_channels
latents = pipe.prepare_latents(
    batch_size * num_images_per_prompt,
    num_channels_latents,
    height,
    width,
    text_embeddings.dtype,
    device,
    generator,
    latents,
)

# Visualize initial latents
plot_latents(latents, device, tmp_dir, title="Initial Latents")

# Decode random latents
with torch.no_grad():
    image = pipe.decode_latents(latents)

# Visualize initial decoded latents
fig, ax = plt.subplots(figsize=(8, 6), nrows=1, ncols=1)
image_copy = image[0, :, :, :]
image_copy = np.transpose(image_copy, (1, 0, 2))
print(image_copy.shape)
ax.imshow(image_copy)
ax.axis("off")
fig.suptitle("Initial Decoded Latents")
fig.tight_layout()
plt.savefig(tmp_dir / "Initial Decoded Latents.png", dpi=128)

# Prepare extra step kwargs
eta = 0.0
extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
