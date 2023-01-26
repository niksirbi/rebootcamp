import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

from rebootcamp.utils import get_device, load_prompts, plot_latents

# Get path to the default config file
default_config_path = (
    Path(__name__).parent.parent / "data" / "default_diffusion_config.yaml"
)
default_config_path = default_config_path.resolve()

# Read arguments from config file
with open(default_config_path, "r") as cfg_file:
    cfg_dict = yaml.safe_load(cfg_file)


positive_prompt = "a shining city on a hill"

# Print arguments
for i in range(1, len(sys.argv)):
    print("argument:", i, "value:", sys.argv[i])

# Get positive prompt from command line
positive_prompt = sys.argv[1]
print(f"prompt: {positive_prompt}")

# Read
data_dir = Path(__name__).parent / "data"
data_dir = data_dir.resolve()
tmp_dir = data_dir / "_tmp"

# Read negative prompts from file
negative_prompt = load_prompts(data_dir / "negative_prompts.txt")

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

# Set height and width
height = 720
width = 960
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
seed = 2021
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
print(image_copy.shape)
ax.imshow(image_copy)
ax.axis("off")
fig.suptitle("Initial Decoded Latents")
fig.tight_layout()
plt.savefig(tmp_dir / "Initial Decoded Latents.png", dpi=128)

# Prepare extra step kwargs
eta = 0.0
extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

start_time = time.time()

# Denoising loop (diffusion!)
num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
with torch.no_grad():
    for i, t in enumerate(timesteps):

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2)
            if do_classifier_free_guidance
            else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(
            latent_model_input, t
        )

        # predict the noise residual
        # Inputs:
        #  - latents *96x96), t (timestep), hidden_states ()
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        # perform guidance
        if do_classifier_free_guidance:

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # UNet predicted noise in text and image embedding
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            # Steps the latents...impements diffusion algorithm
            # Moves latents through latent space towards the
            # manifold of valid images, guided by the text embedding
            latents = pipe.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        else:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond
            # Just denoise initial random latents
            latents = pipe.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # Report progress
        print(f"Iteration {i} of {len(timesteps) - num_warmup_steps}...")

        # Decode and save image every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Iteration {i} of {len(timesteps) - num_warmup_steps}...")
            # Decode current latents
            image = pipe.decode_latents(latents)
            # Save current image
            image = pipe.numpy_to_pil(image)
            image_save_path = tmp_dir / f"step_{str(i + 1).zfill(2)}.png"
            image[0].save(image_save_path)

# Report time
end_time = time.time()
print(f"Diffusion took: {end_time - start_time} seconds")

# Report Done
print("Done")
