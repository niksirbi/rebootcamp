import cv2
import nltk
import numpy as np
import pyaudio
import torch
from diffusers import StableDiffusionPipeline, schedulers
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from rebootcamp.capture import Stream
from rebootcamp.config import FAST_DIFF_CONFIG, DiffusionConfig
from rebootcamp.utils import load_prompts

nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")


# Define subtitle drawing function
def draw_subtitles(image, text, height, color):

    # Specify subtitle font
    font = cv2.FONT_HERSHEY_COMPLEX
    left_position = 20
    bottom_position = height
    fontScale = 1
    fontColor = color
    thickness = 1
    lineType = 2

    bottomLeftCornerOfText = (left_position, bottom_position)

    # Draw subtitles on image
    image = cv2.putText(
        image,
        text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )

    return


# Intialize the config object and set the seed
cfg = DiffusionConfig(FAST_DIFF_CONFIG)
cfg.update({"seed": 2023})

for key, value in cfg.items():
    print(f"{key}: {value}")


def init_diffusion(cfg_dict: DiffusionConfig):
    """
    Initialize the diffusion model.

    Parameters
    ----------
    cfg_dict : DiffusionConfig
        The config dictionary.

    Returns
    -------
    pipe : The diffusion pipeline.
        The diffusion model.
    """

    # Define the model id
    model_id = cfg_dict["model_id"]

    # Load the requested scheduler
    Scheduler = getattr(schedulers, cfg_dict["scheduler_id"])
    scheduler = Scheduler.from_pretrained(model_id, subfolder="scheduler")

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
    if cfg_dict["attention_slicing"]:
        pipe.enable_attention_slicing()

    return pipe


def diffuse(
    pipe: StableDiffusionPipeline,
    prompt: str,
    cfg_dict: DiffusionConfig = cfg,
    seed: int = 42,
):
    """
    Diffuse a prompt.
    """

    # Define call parameters
    batch_size = cfg_dict["batch_size"]
    device = pipe._execution_device
    guidance_scale = cfg_dict["guidance_scale"]

    # Read negative prompts from file
    negative_prompt = load_prompts(cfg_dict["negative_prompts_path"])

    # Encode input text prompt
    num_images_per_prompt = 1
    text_embeddings = pipe._encode_prompt(
        prompt,
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
    generator = torch.Generator(device).manual_seed(seed)

    # Prepare latent variables (random initialize)
    latents = None
    num_channels_latents = pipe.unet.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        cfg_dict["image_height"],
        cfg_dict["image_width"],
        text_embeddings.dtype,
        device,
        generator,
        latents,
    )

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

            # Report progress
            print(f"Iteration {i} of {len(timesteps) - num_warmup_steps}...")

        # Decode and save only the final image
        image = pipe.decode_latents(latents)

        return image


# Initialize the diffusion model
pipe = init_diffusion(cfg)

# Specify word extractors
# is_noun = lambda pos: pos[:2] == "NN"
# is_adjective = lambda pos: pos[:2] == "JJ"


def is_noun(pos):
    return pos[:2] == "NN"


def is_adjective(pos):
    return pos[:2] == "JJ"


# Load processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny.en"
)

# Initiliaze audio capture thread
stream = Stream(1600, pyaudio.paInt16, 1, 16000, 2)
stream.start()

# Open display window
cv2.namedWindow("subtitle")
cv2.setWindowProperty(
    "subtitle", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
)

# Loop
num_slots = 4
noun_slots = ["", "", "", ""]
adj_slots = ["", "", "", ""]
bad_nouns = ["i", "oh", "shit", "fuck"]
bad_adjectives = ["i", "okay"]
while True:

    # Check for keboard input
    k = cv2.waitKey(1) & 0xFF

    # Press 'q' to exit
    if k == ord("q"):
        break

    # Read current sound
    sound = stream.read()

    # Extract current sound features
    inputs = processor(sound, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    # Generate IDs
    generated_ids = model.generate(inputs=input_features)

    # Transcribe
    transcription = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    words = transcription.lower()  # Make all lower case

    # Extract nouns and adjectives
    # words = "How are you? I am OK. Do you like my cat? i am not OK"
    # words = 'orange tigers are tired of getting so much sticky milk and
    # thinking about the rest of the night. You are bad.'
    tokenized = nltk.word_tokenize(words)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    adjectives = [
        word for (word, pos) in nltk.pos_tag(tokenized) if is_adjective(pos)
    ]

    # Update nouns/adjectives
    for noun in nouns:
        if (noun not in noun_slots) and (noun not in bad_nouns):
            noun_slots.pop(0)
            noun_slots.append(noun)
    for adj in adjectives:
        if (adj not in adj_slots) and (adj not in bad_adjectives):
            adj_slots.pop(0)
            adj_slots.append(adj)

    # Update prompt
    prompt = ""
    for i in range(num_slots):
        prompt = prompt + ", " + adj_slots[i] + " " + noun_slots[i]

    # Diffuse a prompt
    image = diffuse(pipe, prompt, cfg_dict=cfg, seed=42)
    image = np.squeeze(image)
    frame = cv2.resize(image, (1920, 1080))
    # Covert color to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Create a black image
    # frame = np.zeros((720, 1280, 3), np.uint8)

    # Draw subtitles
    draw_subtitles(frame, transcription, 670, (255, 255, 255))
    draw_subtitles(frame, prompt, 700, (0, 255, 255))

    # Display the image
    cv2.imshow("subtitle", frame)

# Shutdown
cv2.destroyAllWindows()
stream.stop()

# FIN
