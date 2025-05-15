#!/usr/bin/env python3
import os
import shutil
import glob
import argparse

from helper_code import find_records, load_header, get_label

MISSING_LABEL_MSG = "No label is available: are you trying to load the labels from the held-out data?"

def move_on_missing_label(src_dir: str, dst_dir: str):
    # Ensure absolute paths
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    # We need to be in src_dir so that load_header(record) finds files relative to it
    cwd = os.getcwd()
    os.chdir(src_dir)

    try:
        records = find_records(".")  # returns list of "./sub/path/record" or "record"
        for record in records:
            # normalize leading "./"
            record = record.lstrip("./")
            try:
                header = load_header(record)
                label = get_label(header)
                print(f"[OK]     {record}: label = {label!r}")

            except Exception as e:
                if str(e).startswith(MISSING_LABEL_MSG):
                    # Move every file matching this record base name
                    pattern = record + ".*"
                    for src_path in glob.glob(pattern):
                        rel_path = os.path.relpath(src_path, ".")
                        dst_path = os.path.join(dst_dir, rel_path)
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        #shutil.move(src_path, dst_path)
                        print(f"[MOVED]  {rel_path} → {dst_path}")
                else:
                    # Unexpected exception: re-raise
                    raise

    finally:
        os.chdir(cwd)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Move all files for any record under SRC for which "
            "get_label(load_header(record)) raises the “No label is available…” exception."
        )
    )
    p.add_argument("src", help="Source directory containing WFDB records")
    p.add_argument("dst", help="Destination directory for files with missing labels")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    move_on_missing_label(args.src, args.dst)
