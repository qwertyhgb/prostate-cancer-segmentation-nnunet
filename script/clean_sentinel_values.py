import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FILTERED_ROOT = PROJECT_ROOT / "nnunet" / "FilteredData"
TARGET_FILES = ("ADC.nii", "DWI.nii")
DEFAULT_MIN_VALID_VALUE = 0.0


def clean_one_file(path: Path, min_valid_value: float) -> Dict:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj).copy()

    original_min = float(np.min(data))
    original_max = float(np.max(data))
    negative_voxels = int(np.count_nonzero(data < min_valid_value))
    sentinel_neg32768 = int(np.count_nonzero(data == -32768))

    if negative_voxels == 0:
        return {
            "path": str(path),
            "changed": False,
            "original_min": original_min,
            "original_max": original_max,
            "new_min": original_min,
            "new_max": original_max,
            "negative_voxels": 0,
            "sentinel_neg32768": sentinel_neg32768,
        }

    cleaned = np.clip(data, min_valid_value, None).astype(np.float32)
    new_min = float(np.min(cleaned))
    new_max = float(np.max(cleaned))
    output = nib.Nifti1Image(cleaned, image.affine, image.header.copy())
    output.set_data_dtype(np.float32)
    temp_path = path.with_name(f"{path.stem}__tmp{path.suffix}")
    if temp_path.exists():
        temp_path.unlink()
    nib.save(output, str(temp_path))
    image.uncache()
    del output
    del image
    del data
    del cleaned
    gc.collect()
    temp_path.replace(path)

    return {
        "path": str(path),
        "changed": True,
        "original_min": original_min,
        "original_max": original_max,
        "new_min": new_min,
        "new_max": new_max,
        "negative_voxels": negative_voxels,
        "sentinel_neg32768": sentinel_neg32768,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean sentinel/negative values in ADC and DWI files.")
    parser.add_argument("--filtered-root", type=Path, default=DEFAULT_FILTERED_ROOT)
    parser.add_argument("--min-valid-value", type=float, default=DEFAULT_MIN_VALID_VALUE)
    args = parser.parse_args()

    filtered_root = args.filtered_root.resolve()
    rows: List[Dict] = []

    for filename in TARGET_FILES:
        for path in sorted(filtered_root.glob(f"*/*/{filename}")):
            rows.append(clean_one_file(path, args.min_valid_value))

    changed_rows = [row for row in rows if row["changed"]]
    summary = {
        "filtered_root": str(filtered_root),
        "target_files": list(TARGET_FILES),
        "min_valid_value": args.min_valid_value,
        "total_files": len(rows),
        "changed_files": len(changed_rows),
        "files_with_neg32768": sum(1 for row in rows if row["sentinel_neg32768"] > 0),
        "total_negative_voxels_clipped": int(sum(row["negative_voxels"] for row in changed_rows)),
        "changed_examples": changed_rows[:20],
    }

    with open(filtered_root / "sentinel_cleanup_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"Scanned files: {len(rows)}")
    print(f"Changed files: {len(changed_rows)}")
    print(f"Files with -32768 sentinel values: {summary['files_with_neg32768']}")
    print(f"Total clipped negative voxels: {summary['total_negative_voxels_clipped']}")


if __name__ == "__main__":
    main()
