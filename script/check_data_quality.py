import argparse
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "BPH-PCA"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_quality"

CATEGORIES = ("BPH", "PCA")
REQUIRED_MODALITIES = ("ADC", "DWI", "T2 fs", "T2 not fs")
OPTIONAL_MODALITIES = ("gaoqing-T2",)
ALL_MODALITIES = REQUIRED_MODALITIES + OPTIONAL_MODALITIES


class IssueType(Enum):
    MISSING_ROI = "missing_roi"
    MISSING_MODALITY = "missing_modality"
    LOAD_ERROR_ROI = "load_error_roi"
    LOAD_ERROR_MODALITY = "load_error_modality"
    EMPTY_ROI = "empty_roi"
    MULTI_LABEL_ROI = "multi_label_roi"
    NAN_INF_ROI = "nan_inf_roi"
    NAN_INF_MODALITY = "nan_inf_modality"
    ALL_ZEROS_MODALITY = "all_zeros_modality"
    CONSTANT_MODALITY = "constant_modality"
    SHAPE_MISMATCH_MODALITY = "shape_mismatch_modality"
    SHAPE_MISMATCH_ROI = "shape_mismatch_roi"
    SPACING_MISMATCH_MODALITY = "spacing_mismatch_modality"
    SPACING_MISMATCH_ROI = "spacing_mismatch_roi"
    DIRECTION_MISMATCH_MODALITY = "direction_mismatch_modality"
    DIRECTION_MISMATCH_ROI = "direction_mismatch_roi"


def round_tuple(values: Iterable[float], digits: int = 4) -> Tuple[float, ...]:
    return tuple(round(float(v), digits) for v in values)


def find_roi_root(data_root: Path) -> Path:
    candidates = []
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        if all((child / category).is_dir() for category in CATEGORIES):
            if "roi" in child.name.lower():
                return child
            candidates.append(child)

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot locate ROI directory under {data_root}. "
        f"Expected one directory containing subfolders: {', '.join(CATEGORIES)}."
    )


def is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def read_image_with_fallback(file_path: Path) -> sitk.Image:
    try:
        return sitk.ReadImage(str(file_path))
    except RuntimeError:
        if is_ascii_path(file_path):
            raise

        suffix = "".join(file_path.suffixes) or file_path.suffix
        safe_name = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "._-") else "_" for ch in file_path.stem)
        temp_dir = Path(tempfile.gettempdir()) / "sitk_ascii_cache"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{safe_name}{suffix}"
        shutil.copy2(file_path, temp_path)
        try:
            return sitk.ReadImage(str(temp_path))
        finally:
            if temp_path.exists():
                temp_path.unlink()


def check_image(file_path: Path, is_roi: bool = False) -> Dict:
    result = {
        "path": str(file_path),
        "exists": True,
        "loadable": False,
        "shape": None,
        "spacing": None,
        "origin": None,
        "direction": None,
        "min": None,
        "max": None,
        "dtype": None,
        "voxel_count": None,
        "foreground_voxel_count": None,
        "has_nan": False,
        "has_inf": False,
        "is_constant": False,
        "is_all_zeros": False,
        "unique_values": None,
        "error": None,
        "error_type": None,
    }

    if not file_path.exists():
        result["exists"] = False
        result["error_type"] = "file_not_found"
        return result

    try:
        image = read_image_with_fallback(file_path)
        array = sitk.GetArrayFromImage(image)

        result["loadable"] = True
        result["shape"] = tuple(int(v) for v in array.shape)
        result["spacing"] = round_tuple(image.GetSpacing())
        result["origin"] = round_tuple(image.GetOrigin())
        result["direction"] = round_tuple(image.GetDirection())
        result["dtype"] = str(array.dtype)
        result["voxel_count"] = int(array.size)

        result["has_nan"] = bool(np.isnan(array).any())
        result["has_inf"] = bool(np.isinf(array).any())

        valid_array = array[np.isfinite(array)]
        if valid_array.size == 0:
            result["is_all_zeros"] = True
            result["error_type"] = "empty_array"
            return result

        result["min"] = float(np.min(valid_array))
        result["max"] = float(np.max(valid_array))
        result["is_constant"] = bool(result["min"] == result["max"])
        result["is_all_zeros"] = bool(np.all(valid_array == 0))

        if result["has_nan"] or result["has_inf"]:
            result["error_type"] = "nan_or_inf"
        elif result["is_all_zeros"]:
            result["error_type"] = "all_zeros"
        elif result["is_constant"]:
            result["error_type"] = "constant_value"

        if is_roi:
            unique_values = np.unique(valid_array)
            result["unique_values"] = [float(v) for v in unique_values.tolist()]
            result["foreground_voxel_count"] = int(np.count_nonzero(valid_array))

            if unique_values.size == 0 or np.all(unique_values == 0):
                result["is_all_zeros"] = True
                result["error_type"] = "empty_roi"
            elif unique_values.size > 2:
                result["error_type"] = "multi_label"

    except RuntimeError as exc:
        result["error"] = f"SimpleITK RuntimeError: {exc}"
        result["error_type"] = "load_error"
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"
        result["error_type"] = "unknown_error"

    return result


def add_issue(case_result: Dict, issue_type: IssueType, issue: str) -> None:
    case_result["issues"].append(issue)
    case_result["issue_types"].append(issue_type.value)


def process_case(args: Tuple[str, str, str, Sequence[str], Sequence[str]]) -> Dict:
    category, case_id, data_root_str, required_modalities, optional_modalities = args
    data_root = Path(data_root_str)
    roi_root = find_roi_root(data_root)

    case_result = {
        "category": category,
        "case_id": case_id,
        "modalities": {},
        "roi": None,
        "issues": [],
        "issue_types": [],
    }

    roi_file = roi_root / category / f"{case_id}.nii"
    roi_info = check_image(roi_file, is_roi=True)
    case_result["roi"] = roi_info

    if not roi_info["exists"]:
        add_issue(case_result, IssueType.MISSING_ROI, "Missing ROI file")
    elif not roi_info["loadable"]:
        add_issue(case_result, IssueType.LOAD_ERROR_ROI, f"ROI file cannot be loaded: {roi_info['error']}")
    else:
        if roi_info["is_all_zeros"]:
            add_issue(case_result, IssueType.EMPTY_ROI, "ROI is empty (all zeros)")
        if roi_info["unique_values"] and len(roi_info["unique_values"]) > 2:
            add_issue(case_result, IssueType.MULTI_LABEL_ROI, f"ROI has multiple labels: {roi_info['unique_values']}")
        if roi_info["has_nan"] or roi_info["has_inf"]:
            add_issue(case_result, IssueType.NAN_INF_ROI, "ROI contains NaN or Inf values")

    reference_shape = None
    reference_spacing = None
    reference_direction = None
    reference_modality = None

    for modality in list(required_modalities) + list(optional_modalities):
        modality_file = data_root / category / modality / f"{case_id}.nii"
        modality_info = check_image(modality_file)
        case_result["modalities"][modality] = modality_info

        if not modality_info["exists"]:
            if modality in required_modalities:
                add_issue(case_result, IssueType.MISSING_MODALITY, f"Missing modality: {modality}")
            continue

        if not modality_info["loadable"]:
            add_issue(
                case_result,
                IssueType.LOAD_ERROR_MODALITY,
                f"Modality {modality} cannot be loaded: {modality_info['error']}",
            )
            continue

        if modality_info["is_all_zeros"]:
            add_issue(case_result, IssueType.ALL_ZEROS_MODALITY, f"Modality {modality} is all zeros")
        elif modality_info["is_constant"]:
            add_issue(
                case_result,
                IssueType.CONSTANT_MODALITY,
                f"Modality {modality} is constant (value: {modality_info['min']})",
            )

        if modality_info["has_nan"] or modality_info["has_inf"]:
            add_issue(case_result, IssueType.NAN_INF_MODALITY, f"Modality {modality} contains NaN/Inf values")

        if reference_shape is None:
            reference_shape = modality_info["shape"]
            reference_spacing = modality_info["spacing"]
            reference_direction = modality_info["direction"]
            reference_modality = modality
            continue

        if modality_info["shape"] != reference_shape:
            add_issue(
                case_result,
                IssueType.SHAPE_MISMATCH_MODALITY,
                f"Shape mismatch: {modality} {modality_info['shape']} != {reference_modality} {reference_shape}",
            )
        if modality_info["spacing"] != reference_spacing:
            add_issue(
                case_result,
                IssueType.SPACING_MISMATCH_MODALITY,
                f"Spacing mismatch: {modality} {modality_info['spacing']} != {reference_modality} {reference_spacing}",
            )
        if modality_info["direction"] != reference_direction:
            add_issue(
                case_result,
                IssueType.DIRECTION_MISMATCH_MODALITY,
                f"Direction mismatch: {modality} != {reference_modality}",
            )

    if roi_info["loadable"] and reference_modality is not None:
        if roi_info["shape"] != reference_shape:
            add_issue(
                case_result,
                IssueType.SHAPE_MISMATCH_ROI,
                f"ROI shape mismatch: ROI {roi_info['shape']} != {reference_modality} {reference_shape}",
            )
        if roi_info["spacing"] != reference_spacing:
            add_issue(
                case_result,
                IssueType.SPACING_MISMATCH_ROI,
                f"ROI spacing mismatch: ROI {roi_info['spacing']} != {reference_modality} {reference_spacing}",
            )
        if roi_info["direction"] != reference_direction:
            add_issue(
                case_result,
                IssueType.DIRECTION_MISMATCH_ROI,
                f"ROI direction mismatch: ROI != {reference_modality}",
            )

    return case_result


def collect_cases(data_root: Path, categories: Sequence[str]) -> List[Tuple[str, str]]:
    roi_root = find_roi_root(data_root)
    all_cases = []

    for category in categories:
        roi_dir = roi_root / category
        if not roi_dir.exists():
            print(f"Warning: ROI directory not found for category {category}: {roi_dir}")
            continue

        for roi_file in sorted(roi_dir.glob("*.nii")):
            all_cases.append((category, roi_file.stem))

    return all_cases


def build_summary(results: List[Dict], data_root: Path, output_dir: Path) -> Dict:
    issue_type_counts: Dict[str, int] = {}
    cases_with_issues = 0
    category_counts = {category: 0 for category in CATEGORIES}

    for result in results:
        category_counts[result["category"]] = category_counts.get(result["category"], 0) + 1
        if result["issues"]:
            cases_with_issues += 1
        for issue_type in result["issue_types"]:
            issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1

    sorted_issue_types = sorted(issue_type_counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "total_cases": len(results),
        "cases_with_issues": cases_with_issues,
        "perfect_cases": len(results) - cases_with_issues,
        "category_counts": category_counts,
        "issue_type_counts": dict(sorted_issue_types),
    }


def save_json(data: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_results(results: List[Dict], summary: Dict, output_dir: Path, save_json_enabled: bool, save_csv_enabled: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_json_enabled:
        save_json({"summary": summary, "cases": results}, output_dir / "data_quality_results.json")
        save_json(summary, output_dir / "data_quality_summary.json")

    if save_csv_enabled:
        rows = []
        for result in results:
            row = {
                "case_id": result["case_id"],
                "category": result["category"],
                "has_issues": bool(result["issues"]),
                "num_issues": len(result["issues"]),
                "issues": " | ".join(result["issues"]),
                "issue_types": " | ".join(result["issue_types"]),
            }

            roi = result["roi"]
            row["roi_exists"] = roi["exists"]
            row["roi_loadable"] = roi["loadable"]
            row["roi_shape"] = str(roi["shape"]) if roi["shape"] is not None else "NA"
            row["roi_spacing"] = str(roi["spacing"]) if roi["spacing"] is not None else "NA"
            row["roi_labels"] = str(roi["unique_values"]) if roi["unique_values"] is not None else "NA"
            row["roi_foreground_voxel_count"] = roi["foreground_voxel_count"]

            for modality in ALL_MODALITIES:
                modality_info = result["modalities"].get(modality, {})
                key = modality.replace(" ", "_")
                row[f"{key}_exists"] = modality_info.get("exists", False)
                row[f"{key}_loadable"] = modality_info.get("loadable", False)
                row[f"{key}_shape"] = str(modality_info["shape"]) if modality_info.get("shape") is not None else "NA"
                row[f"{key}_spacing"] = str(modality_info["spacing"]) if modality_info.get("spacing") is not None else "NA"
                row[f"{key}_min"] = modality_info.get("min", "NA")
                row[f"{key}_max"] = modality_info.get("max", "NA")

            rows.append(row)

        pd.DataFrame(rows).to_csv(output_dir / "data_quality_details.csv", index=False, encoding="utf-8-sig")


def load_checkpoint(checkpoint_path: Path) -> List[Dict]:
    if not checkpoint_path.exists():
        return []
    with open(checkpoint_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(results: List[Dict], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def print_summary(summary: Dict) -> None:
    total_cases = summary["total_cases"]
    perfect_cases = summary["perfect_cases"]
    cases_with_issues = summary["cases_with_issues"]

    print("\n" + "=" * 60)
    print("Data quality summary")
    print("=" * 60)
    print(f"Total cases: {total_cases}")
    for category, count in summary["category_counts"].items():
        print(f"  - {category}: {count}")
    if total_cases > 0:
        print(f"Perfect cases: {perfect_cases} ({perfect_cases / total_cases * 100:.2f}%)")
        print(f"Cases with issues: {cases_with_issues} ({cases_with_issues / total_cases * 100:.2f}%)")
    else:
        print("Perfect cases: 0 (0.00%)")
        print("Cases with issues: 0 (0.00%)")

    if summary["issue_type_counts"]:
        print("\nIssue type counts:")
        for issue_type, count in summary["issue_type_counts"].items():
            pct = count / total_cases * 100 if total_cases else 0
            print(f"  - {issue_type}: {count} ({pct:.2f}%)")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the quality of the BPH/PCA multimodal NIfTI dataset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help=f"Dataset root. Default: {DEFAULT_DATA_ROOT}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes. Default: CPU count.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
    parser.add_argument("--no-json", action="store_true", help="Do not write JSON output.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write CSV output.")
    return parser.parse_args()


def run_checks(case_args: List[Tuple[str, str, str, Sequence[str], Sequence[str]]], workers: Optional[int], checkpoint_path: Path) -> List[Dict]:
    results: List[Dict] = []

    max_workers = workers or min(32, max(4, len(case_args)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_case, item): item for item in case_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking"):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0:
                save_checkpoint(results, checkpoint_path)

    return results


def main() -> None:
    args = parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / ".checkpoint_quality_check.json"

    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    roi_root = find_roi_root(data_root)
    print(f"ROI root: {roi_root}")

    print("\nCollecting cases...")
    all_cases = collect_cases(data_root, CATEGORIES)
    if not all_cases:
        raise RuntimeError("No cases found from ROI folders.")

    print(f"Found {len(all_cases)} cases.")

    results: List[Dict] = []
    processed_cases = set()
    if args.resume and checkpoint_path.exists():
        print("Loading checkpoint...")
        results = load_checkpoint(checkpoint_path)
        processed_cases = {(item["category"], item["case_id"]) for item in results}
        all_cases = [(cat, cid) for cat, cid in all_cases if (cat, cid) not in processed_cases]
        print(f"Checkpoint loaded. Remaining cases: {len(all_cases)}")

    if all_cases:
        case_args = [
            (category, case_id, str(data_root), REQUIRED_MODALITIES, OPTIONAL_MODALITIES)
            for category, case_id in all_cases
        ]
        results.extend(run_checks(case_args, args.workers, checkpoint_path))

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    results.sort(key=lambda item: (item["category"], item["case_id"]))
    summary = build_summary(results, data_root, output_dir)

    print("Saving results...")
    save_results(results, summary, output_dir, save_json_enabled=not args.no_json, save_csv_enabled=not args.no_csv)
    print_summary(summary)
    print("Quality check completed.")


if __name__ == "__main__":
    main()
