import os
from pathlib import Path
import json
import click
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration


@click.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False), nargs=1)
@click.option("--fp-col", default="fp", show_default=True,
              help="Name of the CSV column containing image file paths.")
@click.option("--target-dir", required=True, type=click.Path(file_okay=False),
              help="Directory to save generated caption JSON files.")
@click.option("--model-name", default="microsoft/Florence-2-large", show_default=True,
              help="Hugging Face model identifier to use for captioning.")
@click.option("--prompt", default="<CAPTION>", show_default=True,
              help="Task prompt, e.g., <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>.")
@click.option("--max-length", default=64, show_default=True, type=int,
              help="Maximum length of generated captions.")
@click.option("--device", default="cpu", show_default=True,
              help="Torch device for inference, e.g., cpu or cuda.")

def main(csv_path: str, fp_col: str, target_dir: str, model_name: str,
         prompt: str, max_length: int, device: str):
    """
    Generate captions for images using Florence-2 and save each caption
    and image dimensions to a .json file in the target directory
    with the same base name as the image.
    """
    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if fp_col not in df.columns:
        raise click.BadParameter(f"Column '{fp_col}' not found in CSV")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Florence2ForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    for img_path in tqdm(df[fp_col].dropna(), desc="Captioning images"):
        img_file = Path(img_path)
        if not img_file.is_file():
            click.echo(f"Warning: {img_file} not found, skipping.")
            continue

        image = Image.open(img_file).convert("RGB")
        width, height = image.size

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)
        caption = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        data = {
            "prompt": caption,
            "width": width,
            "height": height
        }

        json_path = output_dir / f"{img_file.stem}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(data, jf, ensure_ascii=False, indent=2)

        click.echo(f"JSON saved for {img_file.name} -> {json_path}")

if __name__ == "__main__":
    main()
