import json
from pathlib import Path
import click
import torch
from tqdm import tqdm
from diffusers import SanaPipeline, SanaSprintPipeline
from random import choice
from diffusers.utils.logging import disable_progress_bar
from concurrent.futures import ThreadPoolExecutor

disable_progress_bar()

MODEL_MAP = {
    "600M": "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    "1.6B": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    "Sprint": "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
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
    seed: int,
    device: str
):
    captions_path = Path(captions_dir)
    out_path = Path(target_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_id = MODEL_MAP[model_size]
    click.echo(f"Loading {model_size} model {model_id} on {device}...")

    torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    if model_size == "Sprint":
        pipe = SanaSprintPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    else:
        pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        click.echo(f"Using seed: {seed}")

    with ThreadPoolExecutor() as executor:
        for cap_file in tqdm(captions_path.glob("*.json"), desc="Generating images"):
            data = json.loads(cap_file.read_text(encoding="utf-8"))
            prompt = data.get("prompt")
            if not prompt:
                click.echo(f"Warning: no 'prompt' in {cap_file.name}, skipping.")
                continue
            width = choice([1024] * 6 + [768] * 3 + [512])
            height = choice([1024] * 6 + [768] * 3 + [512])
            pipe.set_progress_bar_config(disable=True)
            output = pipe(
                prompt=prompt,
                height=height,
                width=width,
                generator=generator
            )
            image = output.images[0]
            out_file = out_path / f"{cap_file.stem}.png"
            executor.submit(image.save, out_file)

if __name__ == "__main__":
    main()