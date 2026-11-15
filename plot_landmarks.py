#!/usr/bin/env python3
"""Parse a landmarks_*.txt map file and plot the 3D landmark positions.

File format (lines starting with '#' are comments):
    P <mp_id> <x> <y> <z> <orb_descriptor_hex_256bit>
    O <mp_id> <kf_id> <kf_timestamp_s> <uL> <vL> <uR> <octave> <kp_descriptor_hex_256bit>

Only 'P' lines are needed for positions; 'O' lines are counted per map point
so landmarks can optionally be colored by their number of observations.

Usage:
    python plot_landmarks.py [landmarks_file] [--color-by {obs,z,none}]
                             [--max-points N] [--save out.png]
"""
import argparse
import glob
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def parse_landmarks(path):
    """Return (ids, xyz, descriptors, obs_count) from a landmarks file."""
    ids, xyz, descs = [], [], []
    obs = Counter()
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if line.startswith("P "):
                parts = line.split()
                if len(parts) < 5:
                    print(f"Warning: malformed P line {lineno}", file=sys.stderr)
                    continue
                ids.append(int(parts[1]))
                xyz.append((float(parts[2]), float(parts[3]), float(parts[4])))
                descs.append(parts[5] if len(parts) > 5 else "")
            elif line.startswith("O "):
                obs[int(line.split(maxsplit=2)[1])] += 1
    ids = np.asarray(ids, dtype=np.int64)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    obs_count = np.array([obs.get(i, 0) for i in ids], dtype=np.int64)
    return ids, xyz, descs, obs_count


def quat_to_rot(qx, qy, qz, qw):
    """Unit quaternion (Hamilton, x y z w) -> 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def read_first_kf_pose(path):
    """Read the first pose of a TUM-format keyframe file
    (<t> <tx> <ty> <tz> <qx> <qy> <qz> <qw>). Returns (R, t) of T_w_c."""
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            v = [float(x) for x in line.split()]
            return quat_to_rot(*v[4:8]), np.array(v[1:4])
    raise ValueError(f"No poses found in {path}")


def draw_camera_frame(ax, R, t, length, label="cam"):
    """Draw camera axes: x red, y green, z (optical axis) blue."""
    for i, color in enumerate(("r", "g", "b")):
        d = R[:, i] * length
        ax.quiver(t[0], t[1], t[2], d[0], d[1], d[2], color=color,
                  linewidth=2.5, arrow_length_ratio=0.15)
    ax.text(*(t + 0.1 * length), label, fontsize=9)


def set_equal_aspect(ax, xyz):
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="landmarks txt file (default: first landmark*.txt in cwd)")
    ap.add_argument("--color-by", choices=["obs", "z", "none"], default="obs")
    ap.add_argument("--max-points", type=int, default=0,
                    help="randomly subsample to at most N points (0 = all)")
    ap.add_argument("--point-size", type=float, default=1.0)
    ap.add_argument("--kf", help="TUM keyframe file; camera frame drawn at its first pose "
                                 "(default: first kf*.txt in cwd, else the origin)")
    ap.add_argument("--origin", action="store_true",
                    help="draw the camera frame at (0,0,0) with identity rotation, ignoring --kf")
    ap.add_argument("--axis-length", type=float, default=0.0,
                    help="camera axis length in m (0 = 10%% of the map extent)")
    ap.add_argument("--save", help="save figure to this path instead of only showing")
    args = ap.parse_args()

    path = args.file
    if path is None:
        matches = sorted(glob.glob("landmark*.txt"))
        if not matches:
            sys.exit("No landmark*.txt file found; pass a path explicitly.")
        path = matches[0]

    ids, xyz, descs, obs_count = parse_landmarks(path)
    if len(ids) == 0:
        sys.exit(f"No 'P' landmark lines found in {path}")
    print(f"Loaded {len(ids)} landmarks from {path}")
    print(f"  bounds min: {xyz.min(axis=0)}  max: {xyz.max(axis=0)}")
    print(f"  observations per landmark: mean {obs_count.mean():.1f}, "
          f"min {obs_count.min()}, max {obs_count.max()}")

    if args.max_points and len(ids) > args.max_points:
        sel = np.random.default_rng(0).choice(len(ids), args.max_points, replace=False)
        xyz, obs_count = xyz[sel], obs_count[sel]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    if args.color_by == "obs":
        sc = ax.scatter(*xyz.T, c=obs_count, cmap="viridis", s=args.point_size)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="# observations")
    elif args.color_by == "z":
        sc = ax.scatter(*xyz.T, c=xyz[:, 2], cmap="viridis", s=args.point_size)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="z [m]")
    else:
        ax.scatter(*xyz.T, s=args.point_size, color="tab:blue")

    # Camera frame at the first keyframe (or at the origin)
    R, t, cam_label = np.eye(3), np.zeros(3), "origin"
    kf_path = args.kf
    if kf_path is None and not args.origin:
        kf_matches = sorted(glob.glob("kf*.txt"))
        kf_path = kf_matches[0] if kf_matches else None
    if kf_path and not args.origin:
        R, t = read_first_kf_pose(kf_path)
        cam_label = "KF0"
        print(f"Camera frame at first keyframe of {kf_path}: t = {t}")
    else:
        print("Camera frame at origin")
    length = args.axis_length or 0.10 * (xyz.max(axis=0) - xyz.min(axis=0)).max()
    draw_camera_frame(ax, R, t, length, cam_label)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"{path}  ({len(xyz)} landmarks)")
    set_equal_aspect(ax, xyz)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
