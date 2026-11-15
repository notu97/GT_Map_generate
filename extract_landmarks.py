#!/usr/bin/env python3
"""Extract all 3D landmarks (P lines) and their ORB descriptors from a landmarks_*.txt file.

Input line format:
    P <mp_id> <x> <y> <z> <orb_descriptor_hex_256bit>
(O observation lines and # comments are skipped.)

Output (text, one landmark per line):
    # mp_id x y z orb_descriptor_hex_256bit
    0 -0.274422437 -0.768970430 3.161624908 6c1bfc77...

Optionally also writes a .npz with arrays:
    ids  (N,)      int64
    xyz  (N, 3)    float64
    desc (N, 32)   uint8   (256-bit ORB descriptor as bytes)

Usage:
    python extract_landmarks.py [landmarks_file] [-o out.txt] [--npz out.npz]
"""
import argparse
import glob
import os
import sys

import numpy as np


def extract(path):
    ids, xyz, descs = [], [], []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.startswith("P "):
                continue
            parts = line.split()
            if len(parts) != 6:
                print(f"Warning: skipping malformed P line {lineno}: {line.strip()[:80]}",
                      file=sys.stderr)
                continue
            desc = parts[5].lower()
            if len(desc) != 64:
                print(f"Warning: line {lineno}: descriptor is {len(desc) * 4} bits, "
                      f"expected 256", file=sys.stderr)
            ids.append(int(parts[1]))
            xyz.append((float(parts[2]), float(parts[3]), float(parts[4])))
            descs.append(desc)
    return ids, xyz, descs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?",
                    help="landmarks txt file (default: first landmarks*.txt in cwd)")
    ap.add_argument("-o", "--output",
                    help="output txt (default: <input>_points.txt)")
    ap.add_argument("--npz", help="also save a NumPy .npz (ids, xyz, desc as uint8[32])")
    args = ap.parse_args()

    path = args.file
    if path is None:
        matches = sorted(glob.glob("landmarks*.txt"))
        if not matches:
            sys.exit("No landmarks*.txt file found; pass a path explicitly.")
        path = matches[0]
    out = args.output or os.path.splitext(path)[0] + "_points.txt"
    if os.path.abspath(out) == os.path.abspath(path):
        sys.exit("Output path equals input path; refusing to overwrite.")

    ids, xyz, descs = extract(path)
    if not ids:
        sys.exit(f"No 'P' landmark lines found in {path}")
    if len(set(ids)) != len(ids):
        print(f"Warning: {len(ids) - len(set(ids))} duplicate mp_ids", file=sys.stderr)

    with open(out, "w") as f:
        f.write("# mp_id x y z orb_descriptor_hex_256bit\n")
        for i, (x, y, z), d in zip(ids, xyz, descs):
            f.write(f"{i} {x:.9f} {y:.9f} {z:.9f} {d}\n")
    print(f"Wrote {len(ids)} landmarks to {out}")

    if args.npz:
        desc_bytes = np.array([np.frombuffer(bytes.fromhex(d), dtype=np.uint8) for d in descs])
        np.savez(args.npz, ids=np.asarray(ids, dtype=np.int64),
                 xyz=np.asarray(xyz, dtype=np.float64), desc=desc_bytes)
        print(f"Wrote {args.npz} (ids {len(ids)}, xyz {len(xyz)}x3, desc {desc_bytes.shape})")


if __name__ == "__main__":
    main()
