import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "BPH-PCA"
DEFAULT_QUALITY_RESULTS = PROJECT_ROOT / "outputs" / "data_quality" / "data_quality_results.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "nnunet" / "FilteredData"

INPUT_MODALITIES = ("ADC", "DWI", "T2 not fs")
CATEGORIES = ("BPH", "PCA")


def find_roi_root(data_root: Path) -> Path:
    for child in data_root.iterdir():
        if child.is_dir() and "roi" in child.name.lower():
            if all((child / category).is_dir() for category in CATEGORIES):
                return child
    raise FileNotFoundError(f"ROI directory not found under {data_root}")


def load_quality_cases(results_path: Path) -> List[Dict]:
    with open(results_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload["cases"]


def evaluate_case(case: Dict) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    roi = case["roi"]
    modalities = case["modalities"]

    if not (roi["exists"] and roi["loadable"]):
        reasons.append("roi_unloadable")
        return False, reasons

    if roi.get("is_all_zeros"):
        reasons.append("roi_empty")

    unique_values = roi.get("unique_values")
    if unique_values and len(unique_values) > 2:
        reasons.append("roi_multilabel")

    if roi.get("has_nan") or roi.get("has_inf"):
        reasons.append("roi_naninf")

    reference = None
    for modality in INPUT_MODALITIES:
        info = modalities.get(modality, {})
        if not (info.get("exists") and info.get("loadable")):
            reasons.append(f"{modality}_missing_or_unloadable")
            continue

        if info.get("is_all_zeros"):
            reasons.append(f"{modality}_all_zero")

        if info.get("has_nan") or info.get("has_inf"):
            reasons.append(f"{modality}_naninf")

        shape = info.get("shape")
        if len(shape or ()) != 3:
            reasons.append(f"{modality}_not_3d")

        if reference is None:
            reference = {
                "shape": info.get("shape"),
                "spacing": info.get("spacing"),
                "direction": info.get("direction"),
                "modality": modality,
            }
            continue

        if info.get("shape") != reference["shape"]:
            reasons.append(f"{modality}_shape_mismatch")
        if info.get("spacing") != reference["spacing"]:
            reasons.append(f"{modality}_spacing_mismatch")
        if info.get("direction") != reference["direction"]:
            reasons.append(f"{modality}_direction_mismatch")

    if reference is not None:
        if roi.get("shape") != reference["shape"]:
            reasons.append("roi_shape_mismatch")
        if roi.get("spacing") != reference["spacing"]:
            reasons.append("roi_spacing_mismatch")
        if roi.get("direction") != reference["direction"]:
            reasons.append("roi_direction_mismatch")

    return len(reasons) == 0, reasons


def copy_case_files(data_root: Path, roi_root: Path, output_root: Path, category: str, case_id: str) -> None:
    destination = output_root / category / case_id
    destination.mkdir(parents=True, exist_ok=True)

    file_map = {
        "ADC.nii": data_root / category / "ADC" / f"{case_id}.nii",
        "DWI.nii": data_root / category / "DWI" / f"{case_id}.nii",
        "T2_not_fs.nii": data_root / category / "T2 not fs" / f"{case_id}.nii",
        "ROI.nii": roi_root / category / f"{case_id}.nii",
    }

    for output_name, source in file_map.items():
        if not source.exists():
            raise FileNotFoundError(f"Expected source file missing: {source}")
        shutil.copy2(source, destination / output_name)


def write_summary(output_root: Path, selected_rows: List[Dict], rejected_rows: List[Dict]) -> None:
    summary = {
        "selected_cases": len(selected_rows),
        "rejected_cases": len(rejected_rows),
        "selected_by_category": dict(Counter(row["category"] for row in selected_rows)),
        "rejected_reason_counts": dict(
            Counter(reason for row in rejected_rows for reason in row["reasons"].split(" | "))
        ),
        "modalities_for_training": list(INPUT_MODALITIES),
        "label_file": "ROI.nii",
    }

    with open(output_root / "filter_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter trainable cases for nnUNet lesion segmentation.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--quality-results", type=Path, default=DEFAULT_QUALITY_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean-output", action="store_true", help="Delete existing output root before copying.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    results_path = args.quality_results.resolve()
    output_root = args.output_root.resolve()
    roi_root = find_roi_root(data_root)

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = load_quality_cases(results_path)
    selected_rows: List[Dict] = []
    rejected_rows: List[Dict] = []

    for case in cases:
        ok, reasons = evaluate_case(case)
        row = {
            "category": case["category"],
            "case_id": case["case_id"],
            "reasons": " | ".join(reasons),
        }
        if ok:
            copy_case_files(data_root, roi_root, output_root, case["category"], case["case_id"])
            selected_rows.append(row)
        else:
            rejected_rows.append(row)

    selected_rows.sort(key=lambda item: (item["category"], item["case_id"]))
    rejected_rows.sort(key=lambda item: (item["category"], item["case_id"]))

    pd.DataFrame(selected_rows).to_csv(output_root / "selected_cases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rejected_rows).to_csv(output_root / "rejected_cases.csv", index=False, encoding="utf-8-sig")
    write_summary(output_root, selected_rows, rejected_rows)

    print(f"Selected cases: {len(selected_rows)}")
    print(f"Rejected cases: {len(rejected_rows)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
