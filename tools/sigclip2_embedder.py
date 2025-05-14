#!/usr/bin/env python3
import os
import click
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

@click.group()
@click.option(
    '--model-name',
    default='google/siglip2-base-patch16-256',
    help='HuggingFace SigLIP2 model identifier'
)
@click.option(
    '--device',
    default=('cuda' if torch.cuda.is_available() else 'cpu'),
    help='Device to run the model on (cpu or cuda)'
)
@click.pass_context
def cli(ctx, model_name, device):
    """CLI context: loads processor and model for SigLIP2"""
    click.echo(f'Loading SigLIP2 model `{model_name}` on device `{device}`...')
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    ctx.obj = {
        'processor': processor,
        'model': model,
        'device': device,
        'model_name': model_name
    }


@cli.command()
@click.argument('csv_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--file-col', '-c',
    required=True,
    help='Name of the CSV column containing image file paths'
)
@click.option(
    '--memmap-dir', '-m',
    required=True,
    type=click.Path(),
    help='Directory to save individual memmap files'
)
@click.option(
    '--output-csv', '-o',
    default=None,
    help='Path to save the updated CSV with embedding info'
)
@click.option(
    '--batch-size',
    default=16,
    show_default=True,
    help='Batch size for image processing'
)
@click.pass_context
def image(ctx, csv_path, file_col, memmap_dir, output_csv, batch_size):
    """Extract image embeddings from file paths in CSV and save per-image memmaps"""
    processor = ctx.obj['processor']
    model = ctx.obj['model']
    device = ctx.obj['device']
    model_name = ctx.obj['model_name']

    df = pd.read_csv(csv_path)
    if file_col not in df.columns:
        click.echo(f'Column `{file_col}` not found in {csv_path}')
        return

    os.makedirs(memmap_dir, exist_ok=True)

    paths = df[file_col].astype(str).tolist()
    total = len(paths)
    if total == 0:
        click.echo(f'No file paths found in column `{file_col}`')
        return

    img0 = Image.open(paths[0]).convert('RGB')
    inputs0 = processor(images=[img0], return_tensors='pt').to(device)
    with torch.no_grad():
        emb0 = model.get_image_features(**inputs0)
    dim = emb0.size(-1)

    emb_memmap = [None] * total
    emb_model = [model_name] * total


    for i in tqdm(range(0, total, batch_size), desc='Images'):
        batch_paths = paths[i:i + batch_size]
        imgs = [Image.open(p).convert('RGB') for p in batch_paths]
        inputs = processor(images=imgs, return_tensors='pt').to(device)
        with torch.no_grad():
            embs = model.get_image_features(**inputs).cpu().numpy()

        for j, emb in enumerate(embs):
            idx = i + j
            memmap_path = os.path.join(memmap_dir, f'embedding_{idx}.memmap')
            mmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(dim,))
            mmap[:] = emb
            mmap.flush()
            emb_memmap[idx] = memmap_path


    df['emb_memmap'] = emb_memmap
    df['emb_model'] = emb_model

    if output_csv is None:
        base, ext = os.path.splitext(csv_path)
        output_csv = f"{base}_with_embeddings{ext}"
    df.to_csv(output_csv, index=False)
    click.echo(f'Saved {total} embeddings to {memmap_dir} and updated CSV to {output_csv}')


if __name__ == '__main__':
    cli()
