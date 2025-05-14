#!/usr/bin/env python3
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import pandas as pd
from PIL import Image
from tqdm import tqdm


@click.command()
@click.option(
    '--dir-path', '-d',
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help='Directory containing processed.csv'
)
@click.option(
    '--size', '-s',
    default='224x224',
    help='Resize dimensions in WIDTHxHEIGHT format'
)
@click.option(
    '--quality', '-q',
    default=85,
    type=int,
    show_default=True,
    help='JPEG quality (1-100)'
)
@click.option(
    '--workers', '-w',
    default=16,
    type=int,
    show_default=True,
    help='Number of parallel workers'
)
def main(dir_path, size, quality, workers):
    dir_path = Path(dir_path)
    proc_csv = dir_path / "processed.csv"
    if not proc_csv.exists():
        click.echo(f"Error: {proc_csv} not found", err=True)
        sys.exit(1)

    df = pd.read_csv(proc_csv)
    temp_df = df.copy()
    temp_df["fp"] = temp_df["fp"].apply(
        lambda x: Path(x) if Path(x).is_absolute() else dir_path.parent / Path(x)
    )

    try:
        width, height = map(int, size.lower().split("x"))
    except ValueError:
        click.echo("Error: size must be WIDTHxHEIGHT, e.g. 224x224", err=True)
        sys.exit(1)

    def process_one(fp: Path) -> str:
        if not fp.exists():
            return ""
        parts = list(fp.parts)
        if "images" in parts:
            idx = parts.index("images")
            new_parts = parts[:idx] + ["lowres"] + parts[idx+1:]
            lr_fp = Path(*new_parts)
        else:
            lr_fp = fp.parent.parent / "lowres" / fp.name
        lr_fp.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(fp) as img:
            img = img.resize((width, height), Image.LANCZOS)
            img.save(lr_fp, format="JPEG", quality=quality)
        return str(lr_fp)

    click.echo(f"Starting parallel downsampling ({workers} workers)...")
    lr_fps = [""] * len(df)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(process_one, fp): idx
            for idx, fp in enumerate(temp_df["fp"])
        }
        for fut in tqdm(as_completed(future_to_idx),
                        total=len(future_to_idx),
                        desc="Downsampling",
                        unit="img"):
            idx = future_to_idx[fut]
            try:
                lr_fps[idx] = fut.result()
            except Exception as e:
                tqdm.write(f"Error processing {temp_df['fp'][idx]}: {e}")
                lr_fps[idx] = ""

    df["lr_fp"] = lr_fps
    df.to_csv(proc_csv, index=False)
    click.echo(f"Done: {len(df)} rows written to {proc_csv}")


if __name__ == "__main__":
    main()
