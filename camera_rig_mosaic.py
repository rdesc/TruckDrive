"""Camera-rig mosaic: one synchronized frame from every camera, laid out in the
physical rig arrangement with camera-ID labels.

Reproduces the TruckDrive rig figure -- top row of forward cameras (wide edges
rotated 90 deg into tall peek-strips, three forward views in the middle) and a
bottom row of the sideward/rearward views -- as a clean mosaic with no bboxes,
trajectory, or legend.

Pulls one frame per view straight from S3 (via the instance IAM role) unless a
local scene directory is given. Example::

    python camera_rig_mosaic.py --scene scene_10_5 --frame 40 --out ~/camera_rig.png
    python camera_rig_mosaic.py --local /path/to/scene_10_5 --out rig.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

S3_ROOT = "s3://torc-data/datasets/TruckDrivePublic"

# (view, camera-id) in the physical rig order.
TOP = [("forward_left_wide", 53), ("forward_left_narrow", 51),
       ("forward_center_medium", 65), ("forward_right_narrow", 52),
       ("forward_right_wide", 54)]
BOT = [("rearward_left_bottom_medium", 60), ("sideward_left_back_wide", 56),
       ("sideward_left_front_wide", 55), ("sideward_right_front_wide", 57),
       ("sideward_right_back_wide", 58), ("rearward_right_bottom_medium", 62)]

W = 2400          # canvas width
EW = 150          # top edge (wide) strip width
GAP = 6           # black gutter between the two rows


def _fetch_s3(view: str, scene: str, frame: int, dst: str) -> str:
    """Copy the ``frame``-th sorted image of ``view`` from S3 into ``dst``."""
    prefix = f"{S3_ROOT}/{scene}/camera/leopard/{view}/images/"
    listing = subprocess.run(["aws", "s3", "ls", prefix], capture_output=True,
                             text=True, check=True).stdout.splitlines()
    names = [ln.split()[-1] for ln in listing if ln.strip()]
    name = names[min(frame, len(names) - 1)]
    out = os.path.join(dst, f"{view}.jpg")
    subprocess.run(["aws", "s3", "cp", prefix + name, out, "--quiet"], check=True)
    return out


def _local(view: str, scene_dir: str, frame: int) -> str:
    imgs = sorted(
        os.path.join(scene_dir, "camera", "leopard", view, "images", f)
        for f in os.listdir(os.path.join(scene_dir, "camera", "leopard", view, "images"))
    )
    return imgs[min(frame, len(imgs) - 1)]


def _fit(im: Image.Image, w: int, h: int) -> Image.Image:
    """Center-crop to the target aspect, then resize to (w, h)."""
    iw, ih = im.size
    ta, ia = w / h, iw / ih
    if ia > ta:
        nw = int(ih * ta); x = (iw - nw) // 2
        im = im.crop((x, 0, x + nw, ih))
    else:
        nh = int(iw / ta); y = (ih - nh) // 2
        im = im.crop((0, y, iw, y + nh))
    return im.resize((w, h), Image.LANCZOS)


def _label(im: Image.Image, text: str, fsize: int = 15) -> Image.Image:
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsize)
    except OSError:
        font = ImageFont.load_default()
    tb = d.textbbox((0, 0), text, font=font)
    pad = 4
    d.rectangle([6, 6, 6 + (tb[2] - tb[0]) + 2 * pad, 6 + (tb[3] - tb[1]) + 2 * pad],
                fill=(255, 255, 255))
    d.text((6 + pad, 6 + pad - tb[1]), text, fill=(20, 20, 20), font=font)
    return im


def build(loader, out_path: str) -> str:
    bw = W // 6
    bh = round(bw * 9 / 16)
    tw = (W - 2 * EW) // 3
    th = round(tw * 9 / 16)
    canvas = Image.new("RGB", (W, th + bh + GAP), (0, 0, 0))

    x = 0
    for view, cid in TOP:
        edge = "wide" in view
        w = EW if edge else tw
        im = Image.open(loader(view)).convert("RGB")
        if edge:  # wide edge cameras are shown rotated 90 deg
            im = im.rotate(90 if "left" in view else -90, expand=True)
        canvas.paste(_label(_fit(im, w, th), f"{view}_ID0{cid}", 8 if edge else 15), (x, 0))
        x += w

    x, y = 0, th + GAP
    for view, cid in BOT:
        im = Image.open(loader(view)).convert("RGB")
        canvas.paste(_label(_fit(im, bw, bh), f"{view}_ID0{cid}"), (x, y))
        x += bw

    canvas.save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default="scene_10_5", help="Scene id to pull from S3")
    ap.add_argument("--local", default=None, help="Local staged scene dir (skips S3)")
    ap.add_argument("--frame", type=int, default=40, help="Sorted frame index per view")
    ap.add_argument("--out", default="camera_rig.png")
    args = ap.parse_args()

    if args.local:
        loader = lambda v: _local(v, args.local, args.frame)  # noqa: E731
        print(build(loader, os.path.expanduser(args.out)))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            loader = lambda v: _fetch_s3(v, args.scene, args.frame, tmp)  # noqa: E731
            print(build(loader, os.path.expanduser(args.out)))


if __name__ == "__main__":
    main()
