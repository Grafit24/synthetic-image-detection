import json
from pathlib import Path
import click
import torch
from tqdm import tqdm
from diffusers import SanaPipeline, SanaSprintPipeline

# Mapping of model sizes to Hugging Face repo IDs
MODEL_MAP = {
    "600M": "stabilityai/sd-sana-1024-600m",
    "1.6B": "stabilityai/sd-sana-1024-1.6b",
    "Sprint": "stabilityai/sd-sana-1024-sprint"
}

@click.command()
@click.argument(
    "captions_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    nargs=1
)
@click.option(
    "--target-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory to save generated images (PNG format)."
)
@click.option(
    "--model-size",
    type=click.Choice(list(MODEL_MAP.keys()), case_sensitive=True),
    default="1.6B",
    show_default=True,
    help="Which SANA model to use (1024×1024 versions)."
)
@click.option(
    "--steps",
    default=50,
    show_default=True,
    type=int,
    help="Number of inference steps."
)
@click.option(
    "--cfg-scale",
    default=7.5,
    show_default=True,
    type=float,
    help="Guidance scale (classifier-free guidance)."
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for reproducibility."
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    show_default=True,
    help="Torch device for inference."
)
def main(
    captions_dir: str,
    target_dir: str,
    model_size: str,
    steps: int,
    cfg_scale: float,
    seed: int,
    device: str
):
    """
    Generate images from JSON captions using the SANA pipelines,
    reading width and height from each file, and using bfloat16 on GPU.
    """
    captions_path = Path(captions_dir)
    out_path = Path(target_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Resolve model
    model_id = MODEL_MAP[model_size]
    click.echo(f"Loading {model_size} model {model_id} on {device}...")

    # Determine dtype: use bfloat16 on GPU for performance/quality
    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # Load appropriate pipeline
    if model_size == "Sprint":
        pipe = SanaSprintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype
        )
    else:
        pipe = SanaPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype
        )
    pipe = pipe.to(device)

    # Prepare generator
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        click.echo(f"Using seed: {seed}")

    # Iterate and generate
    for cap_file in tqdm(captions_path.glob("*.json"), desc="Generating images"):
        data = json.loads(cap_file.read_text(encoding="utf-8"))
        prompt = data.get("prompt")
        if not prompt:
            click.echo(f"Warning: no 'prompt' in {cap_file.name}, skipping.")
            continue
        width = data.get("width", 1024)
        height = data.get("height", 1024)

        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg_scale, 
            generator=generator
        )
        image = output.images[0]

        out_file = out_path / f"{cap_file.stem}.png"
        image.save(out_file)
        click.echo(f"Saved image for {cap_file.name} -> {out_file}")

if __name__ == "__main__":
    main()
