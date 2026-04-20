from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from lab_utils.visualization import (
    plot_class_balance,
    plot_numeric_distribution,
)

# Safe project root (works in scripts + notebooks)
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

SEED = 1234
GENERATED_METADATA_PATH = ARTIFACT_DIR / f"lab2_faces_metadata.csv"
 
SPLITS = ("train", "val", "test")
LABELS = ("cat", "dog")
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
 
expected = [DATA_ROOT / split / label for split in SPLITS for label in LABELS]

print(f"Dataset root    : {DATA_ROOT}")
print(f"Metadata path   : {GENERATED_METADATA_PATH}")

def list_image_paths_for_group(data_root: Path, split: str, label: str) -> list[Path]:
    group_dir = data_root / split / label
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(group_dir.glob(ext))
    return sorted(paths)


def inspect_image_file(path: Path) -> tuple[int, int, float]:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return width, height, float(arr.mean())


def make_metadata_row(path: Path, data_root: Path, split: str, label: str) -> dict[str, object]:
    width, height, mean_intensity = inspect_image_file(path)
    return {
        "filepath": str(path.relative_to(data_root)),
        "label": label,
        "split": split,
        "width": width,
        "height": height,
        "mean_intensity": mean_intensity,
    }


def build_metadata_from_folders(data_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for label in LABELS:
            paths = list_image_paths_for_group(data_root, split, label)
            rows.extend(make_metadata_row(p, data_root, split, label) for p in paths)
    return (
        pd.DataFrame(rows)
        .sort_values(["split", "label", "filepath"])
        .reset_index(drop=True)
    )


folder_df = build_metadata_from_folders(DATA_ROOT)
print("metadata shape:", folder_df.shape)
print(folder_df.head())

folder_df.to_csv(GENERATED_METADATA_PATH, index=False)
print(f"Saved metadata to: {GENERATED_METADATA_PATH}")

def load_metadata_table(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


df = load_metadata_table(GENERATED_METADATA_PATH)
print("loaded shape:", df.shape)
print(df.head())

def summarize_metadata(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "class_counts": frame["label"].value_counts(),
        "split_counts": frame["split"].value_counts(),
    }


summary = summarize_metadata(df)
print("Rows    :", summary["rows"])
print("Columns :", summary["columns"])
print("\nClass counts:\n", summary["class_counts"])
print("\nSplit counts:\n", summary["split_counts"])

def build_label_split_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["label", "split"]).size().unstack(fill_value=0)


label_split_table = build_label_split_table(df)
print(label_split_table)

def audit_metadata(frame: pd.DataFrame) -> dict[str, object]:
    allowed = {"cat", "dog"}
    return {
        "missing_values": frame.isnull().sum().to_dict(),
        "duplicate_filepaths": int(frame["filepath"].duplicated().sum()),
        "bad_labels": sorted(set(frame["label"].unique()) - allowed),
        "non_positive_sizes": int(((frame["width"] <= 0) | (frame["height"] <= 0)).sum()),
    }


audit_report = audit_metadata(df)
print(audit_report)

def add_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df_copy = frame.copy()
    df_copy["pixel_count"] = df_copy["width"] * df_copy["height"]
    df_copy["aspect_ratio"] = df_copy["width"] / df_copy["height"]
    ref_size = 64 * 64
    brightness_labels = ["darkest", "dim", "bright", "brightest"]
    try:
        df_copy["brightness_band"] = pd.qcut(
            df_copy["mean_intensity"],
            q=4,
            labels=brightness_labels,
            duplicates="drop",
        )
        if df_copy["brightness_band"].isna().all():
            raise ValueError("all NaN")
    except ValueError:
        df_copy["brightness_band"] = pd.qcut(
            df_copy["mean_intensity"].rank(method="first"),
            q=4,
            labels=brightness_labels,
        )
    df_copy["size_bucket"] = df_copy["pixel_count"].apply(
        lambda pc: "small" if pc < ref_size else ("large" if pc > ref_size else "medium")
    )
    return df_copy


analysis_df = add_analysis_columns(df)
print(analysis_df.head())

def build_split_characteristics_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("split")
        .agg(
            avg_width=("width", "mean"),
            avg_height=("height", "mean"),
            avg_pixel_count=("pixel_count", "mean"),
            avg_mean_intensity=("mean_intensity", "mean"),
        )
    )


split_characteristics = build_split_characteristics_table(analysis_df)
print(split_characteristics)


def sample_balanced_by_split_and_label(
    frame: pd.DataFrame, n_per_group: int, seed: int
) -> pd.DataFrame:
    parts = []
    for _, g in frame.groupby(["split", "label"]):
        parts.append(g.sample(n=min(n_per_group, len(g)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


sample_size_per_group = 5
sampled_df = sample_balanced_by_split_and_label(
    analysis_df, n_per_group=sample_size_per_group, seed=SEED
)
print("sampled shape:", sampled_df.shape)
print(sampled_df.head())

sampled_balance = sampled_df.groupby(["split", "label"]).size().unstack(fill_value=0)
print(sampled_balance)

plot_class_balance(sampled_df, title="Balanced sampled subset by split")
