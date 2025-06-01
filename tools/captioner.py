import os
import json
import click
import pandas as pd
from pathlib import Path, PureWindowsPath
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import pipeline

class ImageDataset(Dataset):
    def __init__(self, csv_path, fp_col):
        self.root = Path(csv_path).parent.parent.parent.parent
        df = pd.read_csv(csv_path)
        if fp_col not in df.columns:
            raise ValueError(f"Column '{fp_col}' not found in CSV")
        self.items = df[fp_col].dropna().tolist()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        raw = PureWindowsPath(self.items[idx]).as_posix()
        path = self.root / raw
        image = Image.open(path).convert("RGB")
        return {"image": image, "stem": path.stem, "name": path.name}

@click.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--fp-col", default="fp", show_default=True)
@click.option("--target-dir", required=True, type=click.Path(file_okay=False))
@click.option("--model-name", default="llava-hf/llava-1.5-7b-hf", show_default=True)
@click.option("--prompt", default="Generate a descriptive caption for this image.", show_default=True)
@click.option("--max-length", default=77, show_default=True, type=int)
@click.option("--num-workers", default=4, show_default=True, type=int)
def main(csv_path, fp_col, target_dir, model_name, prompt, max_length, num_workers):
    output_dir = Path(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    captioner = pipeline(
        "image-text-to-text",
        model=model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        max_new_tokens=max_length,
    )

    dataset = ImageDataset(csv_path, fp_col)
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=lambda batch: batch[0])

    for sample in tqdm(loader, desc="Captioning images"):
        image = sample["image"]
        stem = sample["stem"]

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        result = captioner(text=messages, images=[image])
        caption = result[0]["generated_text"][-1]["content"].strip()

        data = {"prompt": caption, "width": image.width, "height": image.height}
        out_path = output_dir / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(data, jf, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()