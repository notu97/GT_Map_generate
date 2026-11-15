#!/usr/bin/env python3
"""Group landmarks by the keyframe timestamp at which they were observed.

Input (landmarks_*.txt):
    P <mp_id> <x> <y> <z> <orb_descriptor_hex_256bit>
    O <mp_id> <kf_id> <kf_timestamp_s> <uL> <vL> <uR> <octave> <kp_descriptor_hex_256bit>

Output: one block per keyframe timestamp, in chronological order:
    T <kf_timestamp_s> <kf_id> <num_landmarks>
    <mp_id> <x> <y> <z> <uL> <vL> <uR> <octave> <kp_descriptor_hex> <mp_descriptor_hex>
    ...
Landmarks inside a block are sorted by mp_id. uR = -1 means no right-image match.

Usage:
    python landmarks_per_timestamp.py [landmarks_file] [-o out.txt]
"""
import argparse
import glob
import os
import sys
from collections import defaultdict
from decimal import Decimal


def parse(path):
    points = {}                  # mp_id -> (x, y, z, desc)  (strings, kept verbatim)
    obs = defaultdict(list)      # (timestamp_str, kf_id) -> [(mp_id, uL, vL, uR, octave, kp_desc)]
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if line.startswith("P "):
                p = line.split()
                if len(p) != 6:
                    print(f"Warning: malformed P line {lineno}", file=sys.stderr)
                    continue
                points[int(p[1])] = tuple(p[2:6])
            elif line.startswith("O "):
                p = line.split()
                if len(p) != 9:
                    print(f"Warning: malformed O line {lineno}", file=sys.stderr)
                    continue
                obs[(p[3], int(p[2]))].append((int(p[1]), *p[4:9]))
    return points, obs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?",
                    help="landmarks txt file (default: first landmarks*.txt in cwd, "
                         "excluding *_points.txt / *_per_timestamp.txt)")
    ap.add_argument("-o", "--output", help="output txt (default: <input>_per_timestamp.txt)")
    args = ap.parse_args()

    path = args.file
    if path is None:
        matches = sorted(m for m in glob.glob("landmarks*.txt")
                         if not m.endswith(("_points.txt", "_per_timestamp.txt")))
        if not matches:
            sys.exit("No landmarks*.txt file found; pass a path explicitly.")
        path = matches[0]
    out = args.output or os.path.splitext(path)[0] + "_per_timestamp.txt"
    if os.path.abspath(out) == os.path.abspath(path):
        sys.exit("Output path equals input path; refusing to overwrite.")

    points, obs = parse(path)
    if not obs:
        sys.exit(f"No 'O' observation lines found in {path}")

    # Decimal keeps the full nanosecond precision of the timestamp strings when sorting.
    keys = sorted(obs, key=lambda k: (Decimal(k[0]), k[1]))

    missing = 0
    n_obs = 0
    with open(out, "w") as f:
        f.write(f"# Landmarks per keyframe timestamp, from {os.path.basename(path)}\n")
        f.write("# T <kf_timestamp_s> <kf_id> <num_landmarks>\n")
        f.write("# <mp_id> <x> <y> <z> <uL> <vL> <uR> <octave> "
                "<kp_descriptor_hex_256bit> <mp_descriptor_hex_256bit>\n")
        for ts, kf_id in keys:
            rows = sorted(obs[(ts, kf_id)])
            f.write(f"T {ts} {kf_id} {len(rows)}\n")
            for mp_id, uL, vL, uR, octave, kp_desc in rows:
                if mp_id not in points:
                    missing += 1
                    continue
                x, y, z, mp_desc = points[mp_id]
                f.write(f"{mp_id} {x} {y} {z} {uL} {vL} {uR} {octave} {kp_desc} {mp_desc}\n")
                n_obs += 1

    counts = [len(obs[k]) for k in keys]
    print(f"Read {len(points)} landmarks, {sum(counts)} observations from {path}")
    print(f"Wrote {len(keys)} timestamps ({n_obs} landmark rows) to {out}")
    print(f"  {keys[0][0]} -> {keys[-1][0]}; landmarks per timestamp: "
          f"min {min(counts)}, max {max(counts)}, mean {sum(counts) / len(counts):.1f}")
    if missing:
        print(f"Warning: {missing} observations reference mp_ids with no P line (skipped)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
