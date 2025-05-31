import click
import cv2
import numpy as np
import pandas as pd
import concurrent.futures
from PIL import Image, ImageStat
from pathlib import Path
import os
from typing import Callable
from tqdm import tqdm
import imagehash


def get_width(img: Image.Image) -> int:
    return img.width


def get_height(img: Image.Image) -> int:
    return img.height


def get_lum(img: Image.Image) -> float:
    return ImageStat.Stat(img.convert('L')).mean[0]


def is_verified_image(img: Image.Image):
    try:
        img.verify()
        return True
    except:
        return False


def get_hash(img: Image.Image) -> str:
    return str(imagehash.phash(img))


def variance_of_laplacian(img: Image.Image) -> float:
    img = np.array(img.convert('L'))
    return cv2.Laplacian(img, cv2.CV_64F).var()


def process_df(
    df: pd.DataFrame, 
    fp_col: str = 'fp', 
    inplace: bool = False,
    max_workers: int = None,
    **col_and_func: dict[str, Callable]
) -> pd.DataFrame:
    """
    Process a DataFrame of file paths, computing new columns via provided functions.
    """
    def pipe(fp):
        row = {}
        with Image.open(fp) as img:
            for new_col, func in col_and_func.items():
                row[new_col] = func(img)
        return row
    
    data = df if inplace else df.copy()
    paths = data[fp_col].tolist()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(pipe, paths), total=len(paths)))

    new_df = pd.DataFrame(results, index=data.index)
    return pd.concat([data, new_df], axis=1)


def save_csv(df: pd.DataFrame, fp: Path, datasets_dir: Path) -> None:
    """
    Validate and save processed DataFrame to CSV, adjusting file paths to be relative.
    """
    required = {'model_name', 'fp', 'label', 'height', 'width', 'mean_luminance'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    datasets_dir = datasets_dir.absolute()
    df_out = df.copy()
    df_out['fp'] = df_out['fp'].apply(lambda x: str(Path(x).absolute().relative_to(datasets_dir)))
    df_out.to_csv(fp, index=False)


def get_ai_vs_human_data(datasets_dir: Path) -> pd.DataFrame:
    """
    Load AI vs Human dataset and prepare file paths and labels.
    """
    d = datasets_dir / 'ai_vs_human_generated'
    df = pd.read_csv(d / 'train.csv', index_col=0)
    df['fp'] = df['file_name'].apply(lambda x: str(d / x))
    df.drop(["file_name"], axis=1, inplace=True)
    df["model_name"] = np.where(df['label'] == 1, 'SDXL', None)
    return df


def get_sfhq_t2i(datasets_dir: Path) -> pd.DataFrame:
    """
    Load SFHQ T2I dataset, add file paths, model names and labels.
    """
    d = datasets_dir / 'sfhq_t2i'
    df = pd.read_csv(d / 'SFHQ_T2I_dataset.csv')
    df['fp'] = df['image_filename'].apply(lambda x: str(d / 'images' / 'images' / x))
    df.rename(columns={'model_used': 'model_name'}, inplace=True)
    df['label'] = 1
    return df


def parse_sd_faces(datasets_dir: Path) -> pd.DataFrame:
    """
    Parse Stable Diffusion faces dataset into a DataFrame with fp, gender, model_name and label.
    """
    models_map = {'512': 'SD 1.5', '768': 'SD 2.1', '1024': 'SDXL'}
    base = datasets_dir / 'sd_faces' / 'stable-diffusion-face-dataset'
    data = []
    for key, name in models_map.items():
        for gender in ['man', 'woman']:
            for fp in (base / key / gender).glob('*.jpg'):
                data.append({'fp': fp, 'gender': gender,
                             'model_name': name, 'label': 1})
    return pd.DataFrame(data)


def get_openimages(datasets_dir: Path) -> pd.DataFrame:
    """
    Load OpenImages dataset, set labels, remove duplicates by imageid.
    """
    d = datasets_dir / 'openimagesv7'
    df = pd.read_csv(d / 'data.csv')
    df['label'] = 0
    df['model_name'] = None
    df['imageid'] = df['fp'].apply(lambda x: Path(x).stem)
    df.drop_duplicates(subset=['imageid'], inplace=True)
    return df


def get_sana(datasets_dir: Path) -> pd.DataFrame:
    d = datasets_dir / 'sana'
    df = pd.read_csv(d / 'sana_emb.csv')
    df["fp"] = df['fp'].apply(lambda x: str(d / x))
    df['label'] = 1
    df['model_name'] = df["fp"].apply(lambda x: Path(x).parent.name)
    return df


DATASET_FUNCS = {
    'ai_vs_human_generated': get_ai_vs_human_data,
    'sfhq_t2i': get_sfhq_t2i,
    'sd_faces': parse_sd_faces,
    'openimagesv7': get_openimages,
    'sana': get_sana
}

PROCESSED_FUNCS ={
    "height": get_height,
    "width": get_width,
    "mean_luminance": get_lum,
    "sharpness": variance_of_laplacian,
    "hash": get_hash,
    "verified": is_verified_image
}


@click.command()
@click.option(
    '--datasets-dir', '-d', 
    type=click.Path(exists=True, file_okay=False),
    default=lambda: str(Path(os.getcwd()).parent / 'data' / 'datasets'),
    help='Base directory containing dataset subfolders.'
)
@click.option(
    '--dataset', '-n', 
    multiple=True, 
    type=click.Choice(DATASET_FUNCS), 
    default=list(DATASET_FUNCS),
    help='Name(s) of datasets to process.'
)
@click.option(
    '--columns', '-c', 
    multiple=True, 
    type=click.Choice(PROCESSED_FUNCS),
    default=list(PROCESSED_FUNCS),
    help='Columns to compute and add.'
)
@click.option(
    '--max-workers', '-w', 
    type=int, 
    default=None,
    help='Maximum number of worker threads for image processing.'
)
def main(datasets_dir, dataset, columns, max_workers):
    """
    CLI to process and save image dataset statistics.
    """
    datasets_dir = Path(datasets_dir)
    for name in dataset:
        click.echo(f'Processing dataset: {name}')
        get_df = DATASET_FUNCS[name]
        df = get_df(datasets_dir)
        funcs = {col: PROCESSED_FUNCS[col]
                 for col in columns}
        stat_df = process_df(
            df, fp_col='fp', 
            inplace=False, 
            max_workers=max_workers, 
            **funcs
        )

        if 'model_name' not in stat_df.columns:
            stat_df['model_name'] = None
        if 'label' not in stat_df.columns:
            stat_df['label'] = None

        out_fp = datasets_dir / name / 'processed.csv'
        save_csv(stat_df, out_fp, datasets_dir)
        click.echo(f'Saved processed data to: {out_fp}')


if __name__ == '__main__':
    main()
