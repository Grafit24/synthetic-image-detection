#!/usr/bin/env python3
"""
A CLI tool to download images from Open Images V7 by class labels
and generate a CSV of absolute file paths.
"""
import os
import sys
import click
import pandas as pd
from openimages.download import download_images


@click.command()
@click.option(
    '--dir-path', '-d',
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help='Base directory to download images into'
)
@click.option(
    '--class-labels', '-c',
    multiple=True,
    required=True,
    help='One or more class labels to download, e.g., "Human face"'
)
@click.option(
    '--out-csv', '-o',
    type=click.Path(dir_okay=False, writable=True),
    required=True,
    help='Path to output CSV file containing image file paths'
)
def main(dir_path, class_labels, out_csv):
    """
    Downloads images for the specified class labels using openimages,
    then writes a CSV with the absolute paths to all downloaded images.
    """
    # Ensure base directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Create an empty exclusions.txt as required by the API
    exclusions_txt = os.path.join(dir_path, 'exclusions.txt')
    with open(exclusions_txt, 'w') as f:
        pass

    # Download images for each class label
    click.echo(f"Starting download for labels: {', '.join(class_labels)}")
    download_images(dir_path, list(class_labels), exclusions_txt)
    click.echo("Download completed.")

    # Collect all image file paths
    file_paths = []
    for label in class_labels:
        images_dir = os.path.join(dir_path, label, 'images')
        if not os.path.isdir(images_dir):
            click.echo(
                f"Warning: No images directory found for label '{label}'", err=True)
            continue
        for root, _, files in os.walk(images_dir):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, fname)
                    file_paths.append(full_path)

    if not file_paths:
        click.echo("No images found to include in CSV.", err=True)
        sys.exit(1)

    # Write out CSV
    df = pd.DataFrame({'fp': file_paths})
    df.to_csv(out_csv, index=False)
    click.echo(f"CSV with {len(file_paths)} entries written to {out_csv}")


if __name__ == '__main__':
    main()
