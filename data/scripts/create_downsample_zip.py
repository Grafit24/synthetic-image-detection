import os
import csv
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from PIL import Image
from tqdm import tqdm


def _process_image(fp, max_size, base_dir):
    """
    Open the image at fp, downsample so that its largest side <= max_size,
    and return a tuple (arcname, image_bytes).
    """
    try:
        with Image.open(fp) as img:
            orig_format = img.format or 'JPEG'
            width, height = img.size
            # Determine scaling factor
            max_side = max(width, height)
            if max_side > max_size:
                scale = max_size / float(max_side)
                new_size = (int(width * scale), int(height * scale))
                img = img.resize(new_size, Image.LANCZOS)
            # Save to bytes buffer
            buf = io.BytesIO()
            img.save(buf, format=orig_format)
            buf.seek(0)
            # Compute archive name: relative path to base_dir
            rel_path = os.path.relpath(fp, base_dir)
            return rel_path, buf.read()
    except Exception as e:
        # If any image fails, propagate exception
        raise RuntimeError(f"Failed to process image '{fp}': {e}")


@click.command()
@click.argument('csv_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_zip', type=click.Path(dir_okay=False))
@click.option(
    '--max-size',
    default=1024,
    type=int,
    show_default=True,
    help='Maximum size (in pixels) for the largest side of each image.'
)
@click.option(
    '--threads',
    default=16,
    type=int,
    show_default=True,
    help='Number of threads to use for image processing.'
)
def create_downsampled_zip(csv_path, output_zip, max_size, threads):
    """
    Read a CSV file with a column 'fp' containing image file paths.
    Downsample each image so that its largest side is at most MAX_SIZE,
    and package all downsampled images into OUTPUT_ZIP, preserving relative paths.
    Also include the original CSV file at the root of the ZIP.
    """
    base_dir = os.getcwd()

    # Read CSV and collect file paths from 'fp' column
    file_paths = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if 'fp' not in reader.fieldnames:
            raise click.UsageError("CSV must contain a column named 'fp' with file paths.")
        for row in reader:
            fp = row['fp'].strip()
            if not fp:
                continue
            if not os.path.isabs(fp):
                fp_full = os.path.join(base_dir, fp)
            else:
                fp_full = fp
            if not os.path.isfile(fp_full):
                click.echo(f"Warning: File not found, skipping: {fp_full}", err=True)
                continue
            file_paths.append(fp_full)

    if not file_paths:
        click.echo("No valid file paths found in CSV. Exiting.", err=True)
        return

    # Create the ZIP archive
    with zipfile.ZipFile(output_zip, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Add the original CSV at the root of the ZIP
        zf.write(csv_path, arcname=os.path.basename(csv_path))

        # Process images in parallel
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(_process_image, fp, max_size, base_dir): fp
                for fp in file_paths
            }
            for future in tqdm(as_completed(futures), total=len(file_paths)):
                fp = futures[future]
                try:
                    rel_path, img_bytes = future.result()
                    # Ensure directory structure in ZIP
                    zf.writestr(rel_path, img_bytes)
                except Exception as exc:
                    click.echo(f"Error processing {fp}: {exc}", err=True)

    click.echo(f"Created downsampled ZIP archive: {output_zip}")

if __name__ == "__main__":
    create_downsampled_zip()