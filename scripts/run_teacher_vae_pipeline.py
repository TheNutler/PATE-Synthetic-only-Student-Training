#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import timedelta

def run(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print("Command failed:", " ".join(cmd))
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        sys.exit(result.returncode)
    return result

def main():
    p = argparse.ArgumentParser(description="Per-teacher VAE pipeline with progress")
    p.add_argument("--num-teachers", type=int, default=250)
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--pool-size", type=int, default=1000)
    p.add_argument("--target-samples", type=int, default=36)
    p.add_argument("--confidence", type=float, default=0.8)
    p.add_argument("--min-nn-distance", type=float, default=2.0)
    p.add_argument("--min-diversity-distance", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--disable-diversity", action="store_true")
    p.add_argument("--seed-base", type=int, default=123)
    p.add_argument("--teachers-dir", default="wp3_d3.2_saferlearn/trained_nets_gpu")
    p.add_argument("--vaes-dir", default="teacher_vaes")
    p.add_argument("--candidates-dir", default="candidates")
    p.add_argument("--out-teachers-dir", default="teachers")
    p.add_argument("--config", default="config/synthetic_generation.json")
    p.add_argument("--combined-out", default="combined_teacher_vae_dataset")
    args = p.parse_args()

    Path(args.candidates_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_teachers_dir).mkdir(parents=True, exist_ok=True)

    teacher_ids = list(range(args.start_id, args.start_id + args.num_teachers))
    t0 = time.time()
    per_teacher_times = []

    for idx, tid in enumerate(teacher_ids, 1):
        teacher_start = time.time()
        bar_len = 40
        done = int(bar_len * (idx-1) / len(teacher_ids))
        bar = "█" * done + "░" * (bar_len - done)
        elapsed = time.time() - t0
        if per_teacher_times:
            avg = sum(per_teacher_times) / len(per_teacher_times)
            remaining = avg * (len(teacher_ids) - idx + 1)
            eta = str(timedelta(seconds=int(remaining)))
        else:
            eta = "calculating..."
        print(f"\n[{idx}/{len(teacher_ids)}] [{bar}]")
        print(f"Teacher {tid} | Elapsed: {timedelta(seconds=int(elapsed))} | ETA: {eta}")

        decoder = Path(args.vaes_dir) / f"teacher_{tid}" / "decoder.pth"
        candidates = Path(args.candidates_dir) / f"candidates_teacher_{tid}.pt"
        teacher_model = Path(args.teachers_dir) / str(tid) / "model.pth"
        shard_path = Path(args.out_teachers_dir) / f"teacher_{tid}" / "shard.pt"
        meta_path = Path(args.out_teachers_dir) / f"teacher_{tid}" / "metadata.json"
        out_dir = Path(args.out_teachers_dir) / f"teacher_{tid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Generate candidates with teacher-specific VAE
        print("  [1/2] Generating candidates...")
        run([
            sys.executable, "scripts/generate_candidates_teacher_vae.py",
            "--teacher-id", str(tid),
            "--decoder-path", str(decoder),
            "--pool-size", str(args.pool_size),
            "--out", str(candidates),
            "--seed", str(args.seed_base + tid),
            "--config", args.config
        ])

        # 2) Label and filter
        print("  [2/2] Labeling & filtering...")
        cmd = [
            sys.executable, "scripts/label_and_filter.py",
            "--candidates", str(candidates),
            "--teacher-model", str(teacher_model),
            "--config", args.config,
            "--target-samples", str(args.target_samples),
            "--quota", "False",
            "--confidence", str(args.confidence),
            "--min-nn-distance", str(args.min_nn_distance),
            "--min-diversity-distance", str(args.min_diversity_distance),
            "--batch-size", str(args.batch_size),
            "--out-dir", str(out_dir)
        ]
        if shard_path.exists():
            cmd += ["--teacher-shard-path", str(shard_path)]
        if meta_path.exists():
            cmd += ["--shard-metadata", str(meta_path)]
        if args.disable_diversity:
            cmd += ["--disable-diversity"]
        run(cmd)

        tt = time.time() - teacher_start
        per_teacher_times.append(tt)
        print(f"  ✓ Teacher {tid} done in {tt:.1f}s")

    # Combine
    print("\nCombining all teachers...")
    run([
        sys.executable, "scripts/combine_synthetic_datasets.py",
        "--teachers-dir", args.out_teachers_dir,
        "--output", args.combined_out,
        "--percentage", "1.0",
        "--seed", "42"
    ])

    total = time.time() - t0
    print(f"\nAll done. Total time: {timedelta(seconds=int(total))}")
    print(f"Combined dataset: {args.combined_out}")

if __name__ == "__main__":
    main()