"""Download the TruckDrive dataset from Hugging Face.

example: for downloading all cameras + calibrations for scene_28_1
python download_truckdrive.py \
         --scene-ids 28_1 \
         --camera  \
         --calibration \
or 
example: for downloading all modalities for all scenes
python download_truckdrive.py \
         --all-scenes \
         --all-modalities
"""

from __future__ import annotations
import argparse
from pathlib import Path
import zipfile


REPO_ID = "Torc-Robotics/TruckDrive"
MANDATORY_PATTERNS = [
    "CODE_OF_CONDUCT.md",
    "COMMERCIAL-USE-POLICY.txt",
    "LICENSE-COMMERCIAL.txt",
    "LICENSE-NONCOMMERCIAL.txt",
    "NOTICE.txt",
    "README.md",
    "citation_cff.md",
]
OPTIONAL_COMPONENTS = (
    "radar",
    "camera",
    "lidar",
    "poses",
    "calibration",
    "annotations",
    "accumulated_gt_depth",
)
COMPONENT_ARCHIVES = {
    "radar": "radar.zip",
    "camera": "camera.zip",
    "lidar": "lidar.zip",
    "poses": "poses.zip",
    "calibration": "calibrations.zip",
    "annotations": "annotations.zip",
    "accumulated_gt_depth": "accumulated_gt_depth.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the TruckDrive dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./TruckDrive"),
        help="Local directory to store downloaded files.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help="Dataset repo id on Hugging Face Hub, e.g. Torc-Robotics/TruckDrive.",
    )
    parser.add_argument(
        "--scene-ids",
        type=str,
        default=None,
        help="Scene selector. Supports names ('scene_28_1 scene_29_1') or numeric IDs ('28_1 29_1'). Ignored if --all-scenes is set.",
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Download matching modalities for all scenes.",
    )
    parser.add_argument(
        "--radar",
        action="store_true",
        help="Include radar component.",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Include camera component.",
    )
    parser.add_argument(
        "--lidar",
        action="store_true",
        help="Include lidar component.",
    )
    parser.add_argument(
        "--poses",
        action="store_true",
        help="Include poses component.",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Include calibration component.",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        help="Include annotations component.",
    )
    parser.add_argument(
        "--accumulated-gt-depth",
        action="store_true",
        help="Include accumulated ground truth depth component.",
    )
    parser.add_argument(
        "--all-modalities",
        action="store_true",
        help="Include all modality modalities.",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Unzip downloaded modality archives after download.",
    )
    parser.add_argument(
        "--remove-zips-after-unzip",
        action="store_true",
        help="Delete .zip files after successful extraction (requires --unzip).",
    )
    args = parser.parse_args()
    return args


def parse_scene_ids(scene_ids: str | None) -> list[str]:
    if scene_ids is None:
        return []

    tokens = scene_ids.split()
    parsed: list[str] = []
    for token in tokens:
        if token.startswith("scene_"):
            parsed.append(token)
            continue
        elif "_" in token:
            parsed.append(f"scene_{token}")
        else:
            raise ValueError(f"Invalid scene ID token: {token}")

    return parsed


def build_allow_patterns(
    args: argparse.Namespace,
    scene_ids: list[str],
) -> list[str]:
    if args.all_modalities:
        components = list(OPTIONAL_COMPONENTS)
    else:
        components = [
            component
            for component in OPTIONAL_COMPONENTS
            if getattr(args, component, False)
        ]

    if not components:
        raise ValueError(
            "No component selected. Use --all-modalities or one of: "
            "--radar --camera --lidar --poses --calibration --annotations --accumulated-gt-depth"
        )

    patterns: list[str] = [str(pattern) for pattern in MANDATORY_PATTERNS]

    for component in components:
        archive_name = COMPONENT_ARCHIVES[component]
        if args.all_scenes or not scene_ids:
            patterns.append(f"TruckDrive/scene_*/{archive_name}")
        else:
            for scene in scene_ids:
                patterns.append(f"TruckDrive/{scene}/{archive_name}")

    # De-duplicate while preserving order.
    return list(dict.fromkeys(patterns))


def selected_archives(args: argparse.Namespace) -> list[str]:
    if args.all_modalities:
        return [COMPONENT_ARCHIVES[component] for component in OPTIONAL_COMPONENTS]

    selected_components = [
        component
        for component in OPTIONAL_COMPONENTS
        if getattr(args, component, False)
    ]
    return [COMPONENT_ARCHIVES[component] for component in selected_components]


def unzip_downloaded_archives(
    output_dir: Path,
    archives: list[str],
    scene_ids: list[str],
    all_scenes: bool,
    remove_zip_files: bool,
) -> None:
    archive_set = set(archives)
    scene_set = set(scene_ids)

    candidates: list[Path] = []
    for archive_path in output_dir.rglob("*.zip"):
        if archive_path.name not in archive_set:
            continue

        if not all_scenes and scene_set:
            parent_scene = archive_path.parent.name
            if parent_scene not in scene_set:
                continue

        candidates.append(archive_path)

    if not candidates:
        print("No matching zip archives found to unzip.")
        return

    print(f"Unzipping {len(candidates)} archive(s)...")
    for archive_path in sorted(candidates):
        extract_dir = archive_path.with_suffix("")
        extract_dir.mkdir(parents=True, exist_ok=True)

        print(f"[unzip] {archive_path} -> {extract_dir}")
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(path=extract_dir)

        if remove_zip_files:
            archive_path.unlink()
            print(f"[remove] {archive_path}")


def main() -> None:
    args = parse_args()

    if args.remove_zips_after_unzip and not args.unzip:
        raise SystemExit("--remove-zips-after-unzip requires --unzip.")

    try:
        from huggingface_hub import snapshot_download, HfApi

        api = HfApi()
        print(api.whoami())
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_ids = parse_scene_ids(args.scene_ids)

    if args.all_scenes and scene_ids:
        raise SystemExit("Use either --all-scenes or --scene-ids, not both.")

    print(
        "downloading scenes:", "all" if args.all_scenes or not scene_ids else scene_ids
    )

    try:
        allow_patterns = build_allow_patterns(args, scene_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("download patterns", allow_patterns)

    try:
        downloaded_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(args.output_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        if "Repository Not Found" in str(exc):
            raise SystemExit(
                f"Repository not found: {args.repo_id}. "
                "Pass the correct dataset with --repo-id (for example: Torc-Robotics/TruckDrive)."
            ) from exc
        raise

    print(f"Downloaded dataset snapshot to: {downloaded_path}")
    print("Included mandatory patterns: " + ", ".join(MANDATORY_PATTERNS))

    if args.unzip:
        unzip_downloaded_archives(
            output_dir=args.output_dir,
            archives=selected_archives(args),
            scene_ids=scene_ids,
            all_scenes=args.all_scenes,
            remove_zip_files=args.remove_zips_after_unzip,
        )


if __name__ == "__main__":
    main()
