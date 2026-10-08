"""Fetch the chair subset of the 3DShape2VecSet ShapeNet data by HTTP range requests.

occupancies.zip (548 GB) and surfaces.zip (52 GB) are never downloaded whole: remotezip reads
the central directory, then only the requested members.

    python scripts/fetch_data.py lists                 # split lists of the chair category
    python scripts/fetch_data.py shapes --split test   # scale (.npy) + surface (.npz) of every chair in a split
    python scripts/fetch_data.py shapes --ids a b c    # same, for given ids
    python scripts/fetch_data.py occ --ids a b c       # occupancy samples (.npz, ~11 MB each), optional
"""
import argparse
import os
import sys
import time

from remotezip import RemoteZip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CAT = "03001627"
DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
SURFACES_ID = "1Rg_Uyl7DNdX6uKPeQ8g4gm9od6bCWUa5"
OCCUPANCIES_ID = "1dbFfd0YSNItQi6eVzCOLo1m321aM52id"

POINT_DIR = f"ShapeNetV2_point/{CAT}"          # in occupancies.zip
SURF_DIR = f"{CAT}/4_pointcloud"               # in surfaces.zip
LOCAL_POINT = os.path.join(DATA, "ShapeNetV2_point", CAT)
LOCAL_SURF = os.path.join(DATA, "ShapeNetV2_watertight", CAT, "4_pointcloud")


def extract(zip_id, members, dest_of, retries=3):
    todo = [m for m in members if not os.path.exists(dest_of(m))]
    if not todo:
        return
    for attempt in range(retries):
        try:
            with RemoteZip(DRIVE.format(zip_id)) as z:
                for i, m in enumerate(todo):
                    dest = dest_of(m)
                    if os.path.exists(dest):
                        continue
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    tmp = dest + ".part"
                    with z.open(m) as src, open(tmp, "wb") as dst:
                        dst.write(src.read())
                    os.replace(tmp, dest)
                    if (i + 1) % 25 == 0:
                        print(f"  {i + 1}/{len(todo)}", flush=True)
            return
        except Exception as e:  # network hiccups: reopen and skip what is already there
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(5)
    raise RuntimeError("extraction failed")


def read_split(split):
    with open(os.path.join(LOCAL_POINT, f"{split}.lst")) as f:
        return [l.strip().replace(".npz", "") for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["lists", "shapes", "occ"])
    ap.add_argument("--split")
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()

    if a.what == "lists":
        extract(OCCUPANCIES_ID, [f"{POINT_DIR}/{s}.lst" for s in ("train", "val", "test")],
                lambda m: os.path.join(LOCAL_POINT, os.path.basename(m)))
        for s in ("train", "val", "test"):
            print(s, len(read_split(s)))
        return

    ids = a.ids if a.ids else read_split(a.split)
    if a.what == "shapes":
        print(f"scales for {len(ids)} chairs")
        extract(OCCUPANCIES_ID, [f"{POINT_DIR}/{i}.npy" for i in ids],
                lambda m: os.path.join(LOCAL_POINT, os.path.basename(m)))
        print(f"surfaces for {len(ids)} chairs")
        extract(SURFACES_ID, [f"{SURF_DIR}/{i}.npz" for i in ids],
                lambda m: os.path.join(LOCAL_SURF, os.path.basename(m)))
    else:
        extract(OCCUPANCIES_ID, [f"{POINT_DIR}/{i}.npz" for i in ids],
                lambda m: os.path.join(LOCAL_POINT, os.path.basename(m)))


if __name__ == "__main__":
    main()
