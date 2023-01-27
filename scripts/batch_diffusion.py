from itertools import product

from rebootcamp.config import DEFAULT_DIFF_CONFIG, DiffusionConfig
from rebootcamp.diffusion import run_diffusion

# Intialize the config object and set the seed
cfg = DiffusionConfig(DEFAULT_DIFF_CONFIG)
cfg.update({"seed": 100})
cfg.update(
    {"positive_prompt": "cityscape, modern, high definition, cinematic"}
)

# Run diffusion with custom parameters
model_ids = [
    "stabilityai/stable-diffusion-2-1",
    "runwayml/stable-diffusion-v1-5",
]

scheduler_ids = [
    "DDIMScheduler",
    "EulerDiscreteScheduler",
    "PNDMScheduler",
    "LMSDiscreteScheduler",
]

slicing_params = [True, False]
guidance_scales = [5.0, 10.0, 15.0]
possible_steps = [10, 20, 30, 40, 50]


# Iterate both over the model ids and the schedulers
for model_id, scheduler_id, slicing, scale, step in product(
    model_ids, scheduler_ids, slicing_params, guidance_scales, possible_steps
):
    cfg.update({"model_id": model_id})
    cfg.update({"scheduler_id": scheduler_id})
    cfg.update({"attention_slicing": slicing})
    cfg.update({"guidance_scale": scale})
    cfg.update({"num_inference_steps": step})
    try:
        run_diffusion(cfg)
    except Exception:
        print("Failed to run diffusion with the current config.")
        print("Continuing with the next config.")
        continue
