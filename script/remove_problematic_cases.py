import argparse
import json
from pathlib import Path


DEFAULT_CASES = [
    "PCA_0000154016",
    "BPH_0000902786",
]


def remove_case(dataset_dir: Path, case_id: str, dry_run: bool) -> dict:
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    image_files = [images_tr / f"{case_id}_{i:04d}.nii.gz" for i in range(3)]
    label_file = labels_tr / f"{case_id}.nii.gz"

    removed = []
    missing = []

    for path in image_files + [label_file]:
        if path.exists():
            removed.append(str(path))
            if not dry_run:
                path.unlink()
        else:
            missing.append(str(path))

    return {
        "case_id": case_id,
        "removed": removed,
        "missing": missing,
    }


def update_dataset_json(dataset_dir: Path, dry_run: bool) -> int:
    dataset_json = dataset_dir / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset.json not found: {dataset_json}")

    with open(dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_count = len(list((dataset_dir / "labelsTr").glob("*.nii.gz")))
    data["numTraining"] = label_count

    if not dry_run:
        with open(dataset_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return label_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove problematic nnUNet training cases and update dataset.json.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to nnUNet_raw/DatasetXXX_Name",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASES,
        help="Case ids to remove, for example PCA_0000154016 BPH_0000902786",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be removed, do not actually delete files",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    print(f"Dataset dir: {dataset_dir}")
    print(f"Dry run: {args.dry_run}")
    print("Cases to remove:")
    for case_id in args.cases:
        print(f"  - {case_id}")

    results = []
    for case_id in args.cases:
        results.append(remove_case(dataset_dir, case_id, args.dry_run))

    print("\nRemoval summary:")
    for result in results:
        print(f"- {result['case_id']}")
        print(f"  removed: {len(result['removed'])}")
        for path in result["removed"]:
            print(f"    {path}")
        if result["missing"]:
            print(f"  missing: {len(result['missing'])}")
            for path in result["missing"]:
                print(f"    {path}")

    new_count = update_dataset_json(dataset_dir, args.dry_run)
    print(f"\nUpdated numTraining: {new_count}")

    if args.dry_run:
        print("Dry run completed. No files were deleted.")
    else:
        print("Removal completed.")


if __name__ == "__main__":
    main()
