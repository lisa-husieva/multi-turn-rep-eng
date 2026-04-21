"""
Down-select already-saved representation arrays to 8 proportional layers.

Turns (N, N_layers, D) float16 .npy files into (N, 8, D) float16, in place.
Also writes a layer_indices.json next to each processed directory so
downstream code knows which transformer blocks the saved positions map to.

Usage
-----
Dry-run (prints what would be done, no writes):

    python scripts/downselect_layers.py --model llama --dry-run

Actually rewrite the files (atomic per-file: writes to .npy.new then renames):

    python scripts/downselect_layers.py --model llama

Re-running is safe: files already at shape (*, 8, *) are detected and skipped.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def sweep_layers_0_indexed(n_transformer_layers: int, n_sweep: int = 8) -> list[int]:
    """Return 0-indexed positions in the saved array to keep.

    The saved array has shape (N, n_transformer_layers, D) where saved-position k
    corresponds to transformer block k+1 (since the extraction runner drops the
    embedding layer = hidden_states[0]).

    np.linspace(0, n-1, 8) picks 8 evenly-spaced positions including 0 and n-1,
    which in transformer-block labels maps to L1 through L(n_transformer_layers).
    """
    return np.linspace(0, n_transformer_layers - 1, n_sweep, dtype=int).tolist()


def saved_positions_to_transformer_layers(positions_0_indexed: list[int]) -> list[int]:
    """Map 0-indexed saved positions to 1-indexed transformer block numbers."""
    return [p + 1 for p in positions_0_indexed]


def process_array(
    fp: Path,
    keep_positions: list[int],
    dry_run: bool,
) -> tuple[str, tuple, tuple | None]:
    """Process one .npy file. Returns (status, old_shape, new_shape|None)."""
    if not fp.exists():
        return ("missing", (), None)
    arr = np.load(str(fp), mmap_mode="r")
    old_shape = tuple(arr.shape)
    # Already down-selected?
    if arr.ndim == 3 and arr.shape[1] == len(keep_positions):
        return ("skip-already-done", old_shape, old_shape)
    if arr.ndim != 3:
        return ("skip-unexpected-shape", old_shape, None)

    n_layers_in_file = arr.shape[1]
    if max(keep_positions) >= n_layers_in_file:
        return ("skip-not-enough-layers", old_shape, None)

    new_arr = np.asarray(arr[:, keep_positions, :], dtype=np.float16)
    new_shape = tuple(new_arr.shape)

    if dry_run:
        return ("would-rewrite", old_shape, new_shape)

    # Use a temp name ending in .npy so np.save doesn't auto-append another .npy
    tmp = fp.parent / f"{fp.stem}.__tmp__.npy"
    np.save(str(tmp), new_arr)
    assert tmp.exists(), f"expected {tmp} to exist after np.save"
    # Atomic on POSIX — replaces the original in one step
    tmp.replace(fp)
    return ("rewrote", old_shape, new_shape)


def process_directory(
    d: Path,
    n_transformer_layers_hint: int | None,
    dry_run: bool,
    log: list[dict],
) -> None:
    """Process one (trajectories|nocontext|compressed|single_turn)/{subdir}/ directory."""
    # Infer n_transformer_layers from one of the existing arrays if not given
    probe = d / "h_inst.npy"
    if not probe.exists():
        return
    n_layers = np.load(str(probe), mmap_mode="r").shape[1]
    if n_transformer_layers_hint is not None and n_layers != n_transformer_layers_hint:
        # Already down-selected — still write the layer_indices.json if missing
        if n_layers == 8 and not (d / "layer_indices.json").exists():
            # We don't know the original mapping at this point; warn.
            log.append(
                {"dir": str(d), "note": "already 8 layers but no layer_indices.json — write manually"}
            )
        return

    keep_positions = sweep_layers_0_indexed(n_layers, n_sweep=8)
    keep_transformer_layers = saved_positions_to_transformer_layers(keep_positions)

    for fname in ["h_inst.npy", "h_post_inst.npy"]:
        fp = d / fname
        status, old_shape, new_shape = process_array(fp, keep_positions, dry_run)
        log.append(
            {
                "dir": str(d.relative_to(d.parents[2])),
                "file": fname,
                "status": status,
                "old_shape": old_shape,
                "new_shape": new_shape,
            }
        )

    # Write mapping file so downstream code knows which transformer blocks these are
    mapping_path = d / "layer_indices.json"
    mapping = {
        "n_transformer_layers_original": n_layers,
        "n_sweep": 8,
        "saved_positions_0_indexed": keep_positions,
        "transformer_layers_1_indexed": keep_transformer_layers,
        "labels": [f"L{l}" for l in keep_transformer_layers],
    }
    if not dry_run:
        mapping_path.write_text(json.dumps(mapping, indent=2))
    log.append({"dir": str(d.relative_to(d.parents[2])), "file": "layer_indices.json",
                "status": "would-write" if dry_run else "wrote", "mapping": mapping})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="model subdirectory under data/ (e.g. 'llama', 'qwen', 'gemma')")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be done without modifying any files")
    ap.add_argument("--n-transformer-layers", type=int, default=None,
                    help="expected n_layers; arrays that already have fewer are skipped "
                         "(usually auto-detected from the first array)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    root = repo_root / "data" / args.model / "representations"
    if not root.exists():
        print(f"FATAL: {root} does not exist")
        sys.exit(1)

    log: list[dict] = []
    t0 = time.time()

    # trajectories, nocontext, compressed — each has {fw}_{split}/ subdirs
    for cond in ["trajectories", "nocontext", "compressed"]:
        cond_dir = root / cond
        if not cond_dir.exists():
            continue
        for d in sorted(cond_dir.iterdir()):
            if d.is_dir():
                process_directory(d, args.n_transformer_layers, args.dry_run, log)

    # single_turn/{harmful,benign}/
    st_dir = root / "single_turn"
    if st_dir.exists():
        for d in sorted(st_dir.iterdir()):
            if d.is_dir():
                process_directory(d, args.n_transformer_layers, args.dry_run, log)

    elapsed = time.time() - t0

    # Summary
    print("─" * 80)
    print(f"Model:    {args.model}")
    print(f"Mode:     {'DRY RUN' if args.dry_run else 'LIVE REWRITE'}")
    print(f"Elapsed:  {elapsed:.1f}s")
    print("─" * 80)
    status_counts: dict[str, int] = {}
    for row in log:
        status_counts[row.get("status", "?")] = status_counts.get(row.get("status", "?"), 0) + 1
    for st, n in sorted(status_counts.items()):
        print(f"  {st:30s}  {n}")
    print("─" * 80)
    # Detail for rewrites and mappings
    for row in log:
        if row.get("status") in ("rewrote", "would-rewrite"):
            print(f"  {row['dir']}/{row['file']}  {row['old_shape']} → {row['new_shape']}")
        if row.get("file") == "layer_indices.json" and row.get("mapping"):
            m = row["mapping"]
            print(f"  {row['dir']}/layer_indices.json  kept {m['transformer_layers_1_indexed']}")


if __name__ == "__main__":
    main()
