#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import h5py

IO_CHUNK = 1_000_000


def canonical_base(base: str) -> str:
    name = Path(base).name
    if name.endswith(".h5"):
        name = name[:-3]
    if len(name) > 6 and name[-6] == "." and name[-5:].isdigit():
        name = name[:-6]
    return name


def find_parts(sample_dir: Path, base: str):
    stem = canonical_base(base)
    prefix = f"{stem}."
    parts = []
    for p in sample_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not (name.startswith(prefix) and name.endswith(".h5")):
            continue
        mid = name[len(prefix):-3]
        if len(mid) == 5 and mid.isdigit():
            parts.append((int(mid), p))
    parts.sort(key=lambda t: t[0])
    for expected, (idx, _) in enumerate(parts):
        if idx != expected:
            raise RuntimeError(
                f"Missing part index {expected} for base '{stem}' in {sample_dir}"
            )
    if not parts:
        raise RuntimeError(f"No parts found for base '{stem}' in {sample_dir}")
    return parts


def read_shape(path: Path):
    with h5py.File(path, "r") as f:
        sh = f.attrs["shape"]
        return int(sh[0]), int(sh[1])


def copy_parts(src_dir: Path, src_base: str, dst_dir: Path, dst_base: str):
    parts = find_parts(src_dir, src_base)
    dst_stem = canonical_base(dst_base)
    for idx, src in parts:
        dst = dst_dir / f"{dst_stem}.{idx:05d}.h5"
        shutil.copy2(src, dst)


def conj_copy_parts(src_dir: Path, src_base: str, dst_dir: Path, dst_base: str):
    parts = find_parts(src_dir, src_base)
    dst_stem = canonical_base(dst_base)

    for idx, src in parts:
        dst = dst_dir / f"{dst_stem}.{idx:05d}.h5"
        with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
            for key, value in fin.attrs.items():
                fout.attrs[key] = value

            for dset_name in fin.keys():
                din = fin[dset_name]
                dout = fout.create_dataset(
                    dset_name,
                    shape=din.shape,
                    dtype=din.dtype,
                    chunks=din.chunks,
                )

                n = din.shape[0]
                for start in range(0, n, IO_CHUNK):
                    stop = min(start + IO_CHUNK, n)
                    data = din[start:stop]
                    if dset_name == "imag":
                        data = -data
                    dout[start:stop] = data


def check_source(src_dir: Path):
    y_test = find_parts(src_dir, "y_test_file")
    y_sk = find_parts(src_dir, "y_sketch_file")
    z_test = find_parts(src_dir, "z_test_file")
    z_sk = find_parts(src_dir, "z_sketch_file")

    y_test_shape = read_shape(y_test[0][1])
    y_sk_shape = read_shape(y_sk[0][1])
    z_test_shape = read_shape(z_test[0][1])
    z_sk_shape = read_shape(z_sk[0][1])

    if y_test_shape != y_sk_shape:
        raise RuntimeError(
            f"y_test and y_sketch shape mismatch: {y_test_shape} vs {y_sk_shape}"
        )
    if z_test_shape != z_sk_shape:
        raise RuntimeError(
            f"z_test and z_sketch shape mismatch: {z_test_shape} vs {z_sk_shape}"
        )
    if y_test_shape[1] != z_test_shape[1]:
        raise RuntimeError(
            f"column mismatch: y has {y_test_shape[1]} cols, z has {z_test_shape[1]} cols"
        )

    print(
        "Source looks consistent:"
        f" y_rows={y_test_shape[0]}, z_rows={z_test_shape[0]}, dofs={y_test_shape[1]}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="old good sampling directory")
    ap.add_argument("dst", type=Path, help="new sampling directory to create")
    ap.add_argument("--force", action="store_true", help="overwrite dst if it exists")
    args = ap.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.is_dir():
        raise SystemExit(f"Source directory does not exist: {src}")

    if dst.exists():
        if not args.force:
            raise SystemExit(f"Destination exists: {dst} (use --force)")
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    check_source(src)

    print("new_y_test <- old_z_test")
    copy_parts(src, "z_test_file", dst, "y_test_file")

    print("new_y_sketch <- conj(old_z_sketch)")
    conj_copy_parts(src, "z_sketch_file", dst, "y_sketch_file")

    print("new_z_test <- old_y_test")
    copy_parts(src, "y_test_file", dst, "z_test_file")

    print("new_z_sketch <- conj(old_y_sketch)")
    conj_copy_parts(src, "y_sketch_file", dst, "z_sketch_file")

    print(f"Done. Converted cache written to: {dst}")


if __name__ == "__main__":
    main()