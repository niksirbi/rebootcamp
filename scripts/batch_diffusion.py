from rebootcamp.config import DEFAULT_DIFF_CONFIG, DiffusionConfig
from rebootcamp.diffusion import run_diffusion

# Run diffusion with custom parameters
model_ids = [
    "stabilityai/stable-diffusion-2-1",
    "runwayml/stable-diffusion-v1-5",
]

cfg = DiffusionConfig(DEFAULT_DIFF_CONFIG)
cfg.update({"image_height": 960})
cfg.update({"image_width": 720})

for model_id in model_ids:
    cfg.update({"model_id": model_id})
    run_diffusion(cfg)
