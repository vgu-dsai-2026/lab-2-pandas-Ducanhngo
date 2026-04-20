from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from lab_utils.visualization import plot_class_balance, plot_numeric_distribution
SEED = 1234
SPLITS = ('train', 'val', 'test')
LABELS = ('cat', 'dog')
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')

def list_image_paths_for_group(data_root: Path, split: str, label: str) -> list[Path]:
    group_dir = data_root / split / label
    paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(group_dir.glob(ext))
    return sorted(paths)

def inspect_image_file(path: Path) -> tuple[int, int, float]:
    raise NotImplementedError('Inspect one image file.')

def make_metadata_row(path: Path, data_root: Path, split: str, label: str) -> dict[str, object]:
    raise NotImplementedError('Create one metadata row from one image path.')

def build_metadata_from_folders(data_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for label in LABELS:
            paths = list_image_paths_for_group(data_root, split, label)
            rows.extend((make_metadata_row(p, data_root, split, label) for p in paths))
    return pd.DataFrame(rows).sort_values(['split', 'label', 'filepath']).reset_index(drop=True)

def load_metadata_table(csv_path: Path) -> pd.DataFrame:
    raise NotImplementedError('Load the saved metadata table with Pandas.')

def summarize_metadata(frame: pd.DataFrame) -> dict[str, object]:
    raise NotImplementedError('Summarize the metadata table with Pandas.')

def build_label_split_table(frame: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError('Build the label-by-split count table.')

def audit_metadata(frame: pd.DataFrame) -> dict[str, object]:
    raise NotImplementedError('Audit the metadata table.')

def add_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError('Create the analysis columns with Pandas.')

def build_split_characteristics_table(frame: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError('Build the split characteristics summary table.')

def sample_balanced_by_split_and_label(frame: pd.DataFrame, n_per_group: int, seed: int) -> pd.DataFrame:
    raise NotImplementedError('Build a balanced sample across split and label groups.')
sample_size_per_group = 5
