import time
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import torch
import yaml
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

from rebootcamp.config import DiffusionConfig
from rebootcamp.utils import load_prompts, plot_latents

start_time = time.time()

# Define type aliases
MyPath = Union[str, Path]

# Get path to the default config file
default_config_path = (
    Path(__name__).parent.parent / "data" / "default_diffusion_config.yaml"
)
default_config_path = default_config_path.resolve()


def run_diffusion(config_path: MyPath = default_config_path):
    """
    Run the diffusion model given a config file.
    """

    # Read arguments from config file
    with open(default_config_path, "r") as cfg_file:
        cfg_dict = yaml.safe_load(cfg_file)
        cfg_dict = DiffusionConfig(cfg_dict)

    # Print config
    for key, value in cfg_dict.items():
        print(f"{key}: {value}")

    # Define the output directory based on the config
    output_dir = cfg_dict["output_dir"]
    prefix = cfg_dict["run_prefix"]
    # Count how many folderes starting with
    # the prefix are already in the output directory
    num_previous_runs = len(
        [f for f in Path(output_dir).iterdir() if f.name.startswith(prefix)]
    )
    run_index = num_previous_runs + 1
    # Create output directory for this test
    run_dir = Path(output_dir) / f"{prefix}{str(run_index).zfill(3)}"
    print(f"Creating output directory for this run: {run_dir}")
    if not run_dir.exists():
        run_dir.mkdir(parents=True)

    # Get positive prompts from file
    positive_prompt = cfg_dict["positive_prompt"]
    # Read negative prompts from file
    negative_prompt = load_prompts(cfg_dict["negative_prompts_path"])

    # Define the model id
    model_id = cfg_dict["model_id"]

    # Load the requested scheduler
    if cfg_dict["scheduler_id"] == "euler":
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

    # Choose the appropriate dtype
    if cfg_dict["dtype"] == "torch.float16":
        dtype = torch.float16
    elif cfg_dict["dtype"] == "torch.float32":
        dtype = torch.float32
    else:
        raise ValueError(
            "The dtype must be either torch.float16 or torch.float32."
        )

    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, revision="fp16", torch_dtype=dtype
    )
    # Send model to device
    pipe = pipe.to(cfg_dict["device"])

    # Reduces memory usage
    pipe.enable_attention_slicing()

    # Set height and width
    height = cfg_dict["image_height"]
    width = cfg_dict["image_width"]
    # Check inputs. Raise error if not correct
    pipe.check_inputs(positive_prompt, height, width, 1)

    # Define call parameters
    batch_size = cfg_dict["batch_size"]
    device = pipe._execution_device
    guidance_scale = cfg_dict["guidance_scale"]

    # Encode input text prompt
    num_images_per_prompt = 1
    text_embeddings = pipe._encode_prompt(
        positive_prompt,
        device,
        num_images_per_prompt,
        cfg_dict["classifier_free_guidance"],
        negative_prompt,
    )

    # Prepare timesteps
    num_inference_steps = cfg_dict["num_inference_steps"]
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Specify random generator seed
    seed = cfg_dict["seed"]
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

    if cfg_dict["plot_latents"]:
        # Visualize initial latents
        plot_latents(latents, device, run_dir, title="Initial_Latents")

        # Decode random latents
        with torch.no_grad():
            image = pipe.decode_latents(latents)

        # Visualize initial decoded latents
        decoded_title = "Initial_Decoded_Latents"
        fig, ax = plt.subplots(figsize=(8, 6), nrows=1, ncols=1)
        image_copy = image[0, :, :, :]
        print(image_copy.shape)
        ax.imshow(image_copy)
        ax.axis("off")
        fig.suptitle(decoded_title)
        fig.tight_layout()
        plt.savefig(run_dir / decoded_title, dpi=128)

    # Prepare extra step kwargs
    eta = 0.0
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # Denoising loop (diffusion!)
    num_warmup_steps = (
        len(timesteps) - num_inference_steps * pipe.scheduler.order
    )
    with torch.no_grad():
        for i, t in enumerate(timesteps):

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if cfg_dict["classifier_free_guidance"]
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
            if cfg_dict["classifier_free_guidance"]:

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

            # Decode and save only the final image
            if i == len(timesteps) - 1:
                # Decode current latents
                image = pipe.decode_latents(latents)
                # Save current image
                image = pipe.numpy_to_pil(image)
                image_save_path = run_dir / "result.png"
                image[0].save(image_save_path)

    # Report time
    end_time = time.time()
    print(f"Diffusion took: {end_time - start_time} seconds")

    # Add elapsed time to config and save
    cfg_dict["run_duration_sec"] = end_time - start_time
    cfg_dict.dump_to_file(run_dir / "used_config.yaml")

    # Report Done
    print("Done")


run_diffusion()
