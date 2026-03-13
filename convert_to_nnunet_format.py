import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILTERED_ROOT = PROJECT_ROOT / "nnunet" / "FilteredData"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "nnunet" / "nnUNet_raw"
DEFAULT_DATASET_ID = 101
DEFAULT_DATASET_NAME = "Prostate"

CHANNELS = {
    "T2_not_fs.nii": "T2_not_fs",
    "ADC.nii": "ADC",
    "DWI.nii": "DWI",
}
CLIP_NON_NEGATIVE_MODALITIES = {"ADC.nii", "DWI.nii"}


def build_dataset_dir(output_root: Path, dataset_id: int, dataset_name: str) -> Path:
    return output_root / f"Dataset{dataset_id:03d}_{dataset_name}"


def case_identifier(category: str, case_id: str) -> str:
    return f"{category}_{case_id}"


def load_nifti(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def save_image_as_float32(src: Path, dst: Path, force_non_negative: bool = False) -> None:
    image = load_nifti(src)
    data = np.asarray(image.dataobj, dtype=np.float32)
    if force_non_negative:
        data = np.clip(data, 0, None)
    output = nib.Nifti1Image(data, image.affine, image.header.copy())
    output.set_data_dtype(np.float32)
    nib.save(output, str(dst))


def save_label_as_uint8_binary(src: Path, dst: Path) -> None:
    image = load_nifti(src)
    data = np.asarray(image.dataobj)
    binary = (data > 0.5).astype(np.uint8)
    output = nib.Nifti1Image(binary, image.affine, image.header.copy())
    output.set_data_dtype(np.uint8)
    nib.save(output, str(dst))


def collect_cases(filtered_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for category_dir in sorted(filtered_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for case_dir in sorted(category_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            row = {
                "category": category,
                "case_id": case_dir.name,
                "path": case_dir,
            }
            rows.append(row)
    return rows


def write_dataset_json(dataset_dir: Path, dataset_name: str, num_training: int) -> None:
    payload = {
        "channel_names": {
            "0": "T2_not_fs",
            "1": "ADC",
            "2": "DWI",
        },
        "labels": {
            "background": 0,
            "lesion": 1,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "Filtered multimodal prostate lesion segmentation dataset for nnUNet v2",
    }
    with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FilteredData to nnUNet raw dataset format.")
    parser.add_argument("--filtered-root", type=Path, default=DEFAULT_FILTERED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--clean-output", action="store_true", help="Delete target dataset directory before conversion.")
    args = parser.parse_args()

    filtered_root = args.filtered_root.resolve()
    output_root = args.output_root.resolve()
    dataset_dir = build_dataset_dir(output_root, args.dataset_id, args.dataset_name)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    if not filtered_root.exists():
        raise FileNotFoundError(f"Filtered root not found: {filtered_root}")

    if args.clean_output and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(filtered_root)
    if not cases:
        raise RuntimeError(f"No cases found under {filtered_root}")

    summary_rows: List[Dict] = []

    for item in cases:
        category = item["category"]
        case_id = item["case_id"]
        case_dir = item["path"]
        nnunet_case_id = case_identifier(category, case_id)

        for channel_index, filename in enumerate(("T2_not_fs.nii", "ADC.nii", "DWI.nii")):
            src = case_dir / filename
            if not src.exists():
                raise FileNotFoundError(f"Missing input modality file: {src}")
            dst = images_tr / f"{nnunet_case_id}_{channel_index:04d}.nii.gz"
            save_image_as_float32(src, dst, force_non_negative=filename in CLIP_NON_NEGATIVE_MODALITIES)

        label_src = case_dir / "ROI.nii"
        if not label_src.exists():
            raise FileNotFoundError(f"Missing label file: {label_src}")
        label_dst = labels_tr / f"{nnunet_case_id}.nii.gz"
        save_label_as_uint8_binary(label_src, label_dst)

        summary_rows.append(
            {
                "category": category,
                "case_id": case_id,
                "nnunet_case_id": nnunet_case_id,
                "label_path": str(label_dst),
            }
        )

    write_dataset_json(dataset_dir, args.dataset_name, len(cases))
    with open(dataset_dir / "conversion_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_id": args.dataset_id,
                "dataset_name": args.dataset_name,
                "num_cases": len(cases),
                "filtered_root": str(filtered_root),
                "dataset_dir": str(dataset_dir),
                "cases": summary_rows,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Converted cases: {len(cases)}")
    print(f"Dataset dir: {dataset_dir}")


if __name__ == "__main__":
    main()
