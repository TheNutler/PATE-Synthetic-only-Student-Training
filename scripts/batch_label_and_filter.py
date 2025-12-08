#!/usr/bin/env python3
"""Batch process label and filter for multiple teachers.

This script processes multiple teachers sequentially, calling label_and_filter.py
for each teacher model found in the trained_nets_gpu directory.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional
from datetime import timedelta


def find_teacher_models(teachers_dir: Path) -> List[int]:
    """
    Find all teacher model directories.
    
    Args:
        teachers_dir: Directory containing teacher model subdirectories
        
    Returns:
        Sorted list of teacher IDs
    """
    teacher_ids = []
    for item in teachers_dir.iterdir():
        if item.is_dir():
            try:
                teacher_id = int(item.name)
                model_path = item / 'model.pth'
                if model_path.exists():
                    teacher_ids.append(teacher_id)
            except ValueError:
                continue
    
    return sorted(teacher_ids)


def generate_candidate_pool(
    teacher_id: int,
    decoder_path: Path,
    candidates_output_dir: Path,
    pool_size: int,
    latent_dim: int,
    latent_mixing: float,
    latent_noise: float,
    seed: int,
    config_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Generate a candidate pool for a specific teacher.
    
    Args:
        teacher_id: Teacher ID
        decoder_path: Path to pretrained decoder
        candidates_output_dir: Directory to save candidate pools
        pool_size: Size of candidate pool
        latent_dim: Latent dimension
        latent_mixing: Latent mixing ratio
        latent_noise: Latent noise scale
        seed: Random seed (will be offset by teacher_id for uniqueness)
        config_path: Path to config file
        
    Returns:
        Path to generated candidate pool, or None if generation failed
    """
    candidates_output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = candidates_output_dir / f'candidates_teacher_{teacher_id}.pt'
    
    # Use teacher_id to offset seed for unique pools per teacher
    teacher_seed = seed + teacher_id
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'generate_candidates.py'),
        '--decoder-path', str(decoder_path),
        '--pool-size', str(pool_size),
        '--latent-dim', str(latent_dim),
        '--latent-mixing', str(latent_mixing),
        '--latent-noise', str(latent_noise),
        '--out', str(candidates_path),
        '--run-id', f'teacher_{teacher_id}',
        '--seed', str(teacher_seed)
    ]
    
    if config_path and config_path.exists():
        cmd.extend(['--config', str(config_path)])
    
    try:
        # Suppress verbose output during batch processing
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if candidates_path.exists():
            return candidates_path
        else:
            print(f'\n   Warning: Candidate pool not found after generation: {candidates_path}')
            return None
    except subprocess.CalledProcessError as e:
        print(f'\n   Error generating candidate pool: {e.stderr[:200] if e.stderr else "Unknown error"}')
        return None


def process_teacher(
    teacher_id: int,
    candidates_path: Path,
    teachers_base_dir: Path,
    output_base_dir: Path,
    confidence: float,
    quota: str,
    min_nn_distance: float,
    min_diversity_distance: float,
    max_per_class: Optional[int],
    teacher_shard_base: Optional[Path],
    shard_metadata_base: Optional[Path],
    batch_size: int,
    decoder_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    target_samples: Optional[int] = None,
    disable_diversity: bool = False
) -> bool:
    """
    Process a single teacher by calling label_and_filter.py.
    
    Args:
        teacher_id: Teacher ID
        candidates_path: Path to candidate pool
        teachers_base_dir: Base directory for teacher models
        output_base_dir: Base directory for outputs
        confidence: Confidence threshold
        quota: Whether to match class distribution
        min_nn_distance: Minimum NN distance for memorization check
        min_diversity_distance: Minimum diversity distance
        max_per_class: Max samples per class
        teacher_shard_base: Base directory for teacher shards (optional)
        shard_metadata_base: Base directory for shard metadata (optional)
        batch_size: Batch size for inference
        
    Returns:
        True if successful, False otherwise
    """
    teacher_model_path = teachers_base_dir / str(teacher_id) / 'model.pth'
    output_dir = output_base_dir / f'teacher_{teacher_id}'
    
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'label_and_filter.py'),
        '--candidates', str(candidates_path),
        '--teacher-model', str(teacher_model_path),
        '--confidence', str(confidence),
        '--quota', quota,
        '--min-nn-distance', str(min_nn_distance),
        '--min-diversity-distance', str(min_diversity_distance),
        '--out-dir', str(output_dir),
        '--batch-size', str(batch_size)
    ]
    
    # Add optional shard path
    if teacher_shard_base:
        shard_path = teacher_shard_base / f'teacher_{teacher_id}' / 'shard.pt'
        if shard_path.exists():
            cmd.extend(['--teacher-shard-path', str(shard_path)])
    
    # Add optional metadata
    metadata_found = False
    if shard_metadata_base:
        metadata_path = shard_metadata_base / f'teacher_{teacher_id}' / 'metadata.json'
        if metadata_path.exists():
            cmd.extend(['--shard-metadata', str(metadata_path)])
            metadata_found = True
    
    if not metadata_found and quota.lower() == 'true':
        # Try default location
        metadata_path = output_base_dir / f'teacher_{teacher_id}' / 'metadata.json'
        if metadata_path.exists():
            cmd.extend(['--shard-metadata', str(metadata_path)])
            metadata_found = True
    
    # If quota is True but no metadata found, set quota to False to avoid errors
    if quota.lower() == 'true' and not metadata_found and not max_per_class:
        # Remove quota flag and add it as False, or use max_per_class as fallback
        # Actually, the label_and_filter script will handle this gracefully now
        pass
    
    # Add max-per-class if specified
    if max_per_class:
        cmd.extend(['--max-per-class', str(max_per_class)])
    
    # Add decoder path if specified (for latent steering)
    if decoder_path and decoder_path.exists():
        cmd.extend(['--decoder-path', str(decoder_path)])
    
    # Add config path if specified
    if config_path and config_path.exists():
        cmd.extend(['--config', str(config_path)])
    
    # Add target samples if specified
    if target_samples is not None:
        cmd.extend(['--target-samples', str(target_samples)])
    
    # Add disable diversity flag if specified
    if disable_diversity:
        cmd.append('--disable-diversity')
    
    # Run command (capture output to reduce noise)
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Only show errors if they occur, not full output
        return True
    except subprocess.CalledProcessError as e:
        # Print error details on failure
        if e.stderr:
            print(f'\n   Error: {e.stderr[:200]}')
        if e.stdout:
            # Print last few lines of stdout for context
            output_lines = e.stdout.strip().split('\n')
            if len(output_lines) > 3:
                print(f'   Last output:')
                for line in output_lines[-3:]:
                    print(f'     {line}')
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch process label and filter for multiple teachers'
    )
    parser.add_argument(
        '--candidates',
        type=str,
        default=None,
        help='Path to shared candidate pool .pt file (if not generating per teacher)'
    )
    parser.add_argument(
        '--generate-candidates-per-teacher',
        action='store_true',
        help='Generate a separate candidate pool for each teacher (requires --decoder-path)'
    )
    parser.add_argument(
        '--teachers-dir',
        type=str,
        default='wp3_d3.2_saferlearn/trained_nets_gpu',
        help='Directory containing teacher models (default: wp3_d3.2_saferlearn/trained_nets_gpu)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='teachers',
        help='Base output directory for synthetic datasets (default: teachers)'
    )
    parser.add_argument(
        '--num-teachers',
        type=int,
        default=None,
        help='Number of teachers to process (default: all available)'
    )
    parser.add_argument(
        '--start-id',
        type=int,
        default=0,
        help='Starting teacher ID (default: 0)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.9,
        help='Confidence threshold (default: 0.9)'
    )
    parser.add_argument(
        '--quota',
        type=str,
        default='True',
        help='Match class distribution from metadata (default: True)'
    )
    parser.add_argument(
        '--min-nn-distance',
        type=float,
        default=1.0,
        help='Minimum NN distance for memorization check (default: 1.0)'
    )
    parser.add_argument(
        '--min-diversity-distance',
        type=float,
        default=0.0,
        help='Minimum diversity distance (default: 0.0, disabled)'
    )
    parser.add_argument(
        '--max-per-class',
        type=int,
        default=None,
        help='Maximum samples per class (default: None)'
    )
    parser.add_argument(
        '--teacher-shard-base',
        type=str,
        default=None,
        help='Base directory for teacher shard images (optional)'
    )
    parser.add_argument(
        '--shard-metadata-base',
        type=str,
        default=None,
        help='Base directory for shard metadata JSON files (optional)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for inference (default: 128)'
    )
    parser.add_argument(
        '--decoder-path',
        type=str,
        default=None,
        help='Path to pretrained decoder.pth (required for --generate-candidates-per-teacher, optional for latent steering)'
    )
    parser.add_argument(
        '--pool-size',
        type=int,
        default=None,
        help='Size of candidate pool per teacher (default: 20000 from config, only used with --generate-candidates-per-teacher)'
    )
    parser.add_argument(
        '--candidates-dir',
        type=str,
        default='candidates',
        help='Directory to save teacher-specific candidate pools (default: candidates)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: config/synthetic_generation.json)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Random seed for candidate generation (default: 123, will be offset by teacher_id)'
    )
    parser.add_argument(
        '--target-samples',
        type=int,
        default=None,
        help='Target number of samples to select per teacher (selects top N by confidence after all filtering)'
    )
    parser.add_argument(
        '--disable-diversity',
        action='store_true',
        help='Disable diversity filter entirely for maximum speed (not recommended for quality)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    candidates_path = Path(args.candidates) if args.candidates else None
    teachers_dir = Path(args.teachers_dir)
    output_dir = Path(args.output_dir)
    teacher_shard_base = Path(args.teacher_shard_base) if args.teacher_shard_base else None
    shard_metadata_base = Path(args.shard_metadata_base) if args.shard_metadata_base else None
    decoder_path = Path(args.decoder_path) if args.decoder_path else None
    config_path = Path(args.config) if args.config else None
    candidates_dir = Path(args.candidates_dir)
    
    # Load config for defaults
    pool_size = args.pool_size
    latent_dim = 32
    latent_mixing = 0.3
    latent_noise = 0.1
    
    if config_path and config_path.exists():
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            gen_config = config.get('generation', {})
            if pool_size is None:
                pool_size = gen_config.get('pool_size', 20000)
            latent_dim = gen_config.get('latent_dim', 32)
            latent_mixing = gen_config.get('latent_mixing_ratio', 0.3)
            latent_noise = gen_config.get('latent_noise_scale', 0.1)
        except Exception as e:
            print(f'Warning: Could not load config: {e}')
    
    if pool_size is None:
        pool_size = 20000
    
    # Validate inputs
    if args.generate_candidates_per_teacher:
        if decoder_path is None or not decoder_path.exists():
            print(f'❌ Error: --decoder-path is required when using --generate-candidates-per-teacher')
            print(f'   Decoder path: {decoder_path}')
            sys.exit(1)
        print(f'✓ Will generate candidate pool per teacher (pool_size={pool_size})')
    else:
        if candidates_path is None or not candidates_path.exists():
            print(f'❌ Error: Candidate pool not found: {candidates_path}')
            print(f'   Use --generate-candidates-per-teacher to generate pools per teacher, or provide --candidates')
            sys.exit(1)
        print(f'✓ Using shared candidate pool: {candidates_path}')
    
    if not teachers_dir.exists():
        print(f'❌ Error: Teachers directory not found: {teachers_dir}')
        sys.exit(1)
    
    # Find all teacher models
    print(f'Scanning for teacher models in: {teachers_dir}')
    teacher_ids = find_teacher_models(teachers_dir)
    
    if not teacher_ids:
        print(f'❌ Error: No teacher models found in {teachers_dir}')
        sys.exit(1)
    
    print(f'Found {len(teacher_ids)} teacher models')
    
    # Filter by start_id and num_teachers
    if args.start_id > 0:
        teacher_ids = [tid for tid in teacher_ids if tid >= args.start_id]
    
    if args.num_teachers:
        teacher_ids = teacher_ids[:args.num_teachers]
    
    print(f'Processing {len(teacher_ids)} teachers (IDs: {teacher_ids[0]} to {teacher_ids[-1]})')
    print(f'Output directory: {output_dir}')
    print(f'Confidence threshold: {args.confidence}')
    print(f'Quota matching: {args.quota}')
    print(f'Min NN distance: {args.min_nn_distance}')
    if args.generate_candidates_per_teacher:
        print(f'Candidate pool per teacher: {pool_size} images')
        print(f'Candidates directory: {candidates_dir}')
    print('=' * 60)
    
    # Process each teacher
    successful = 0
    failed = 0
    start_time = time.time()
    teacher_times = []
    
    print('\n' + '=' * 60)
    print('Starting batch processing...')
    print('=' * 60)
    
    for i, teacher_id in enumerate(teacher_ids, 1):
        teacher_start_time = time.time()
        
        # Progress bar
        progress_pct = (i - 1) / len(teacher_ids) * 100
        bar_length = 40
        filled = int(bar_length * (i - 1) / len(teacher_ids))
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Estimated time remaining
        if i > 1 and len(teacher_times) > 0:
            avg_time = sum(teacher_times) / len(teacher_times)
            remaining = avg_time * (len(teacher_ids) - i + 1)
            remaining_str = str(timedelta(seconds=int(remaining)))
        else:
            remaining_str = "calculating..."
        
        # Elapsed time
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print(f'\n[{i}/{len(teacher_ids)}] [{bar}] {progress_pct:.1f}%')
        print(f'Processing teacher {teacher_id}...')
        print(f'Elapsed: {elapsed_str} | Est. remaining: {remaining_str}')
        
        # Generate candidate pool for this teacher if requested
        if args.generate_candidates_per_teacher:
            gen_start = time.time()
            print(f'  [1/2] Generating candidate pool ({pool_size} images)...', end=' ', flush=True)
            teacher_candidates_path = generate_candidate_pool(
                teacher_id,
                decoder_path,
                candidates_dir,
                pool_size,
                latent_dim,
                latent_mixing,
                latent_noise,
                args.seed if hasattr(args, 'seed') else 123,
                config_path
            )
            gen_time = time.time() - gen_start
            
            if teacher_candidates_path is None:
                print(f'❌ Failed ({gen_time:.1f}s)')
                failed += 1
                continue
            
            print(f'✓ Done ({gen_time:.1f}s)')
            current_candidates_path = teacher_candidates_path
        else:
            current_candidates_path = candidates_path
        
        # Label and filter
        label_start = time.time()
        step_num = "2/2" if args.generate_candidates_per_teacher else "1/1"
        print(f'  [{step_num}] Labeling and filtering...', end=' ', flush=True)
        
        success = process_teacher(
            teacher_id,
            current_candidates_path,
            teachers_dir,
            output_dir,
            args.confidence,
            args.quota,
            args.min_nn_distance,
            args.min_diversity_distance,
            args.max_per_class,
            teacher_shard_base,
            shard_metadata_base,
            args.batch_size,
            decoder_path,
            config_path,
            args.target_samples,
            args.disable_diversity
        )
        
        label_time = time.time() - label_start
        teacher_total_time = time.time() - teacher_start_time
        
        if success:
            successful += 1
            status = '✓'
        else:
            failed += 1
            status = '❌'
        
        print(f'{status} Done ({label_time:.1f}s)')
        print(f'  Teacher {teacher_id} total time: {teacher_total_time:.1f}s')
        
        teacher_times.append(teacher_total_time)
        
        # Update progress bar
        progress_pct = i / len(teacher_ids) * 100
        filled = int(bar_length * i / len(teacher_ids))
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'  Progress: [{bar}] {progress_pct:.1f}%')
    
    # Summary
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    avg_time = sum(teacher_times) / len(teacher_times) if teacher_times else 0
    
    print('\n' + '=' * 60)
    print('Batch processing complete!')
    print('=' * 60)
    print(f'Results:')
    print(f'  Successful: {successful}/{len(teacher_ids)} ({successful/len(teacher_ids)*100:.1f}%)')
    print(f'  Failed: {failed}/{len(teacher_ids)} ({failed/len(teacher_ids)*100:.1f}%)')
    print(f'\nTiming:')
    print(f'  Total time: {total_time_str}')
    print(f'  Average per teacher: {avg_time:.1f}s')
    if teacher_times:
        print(f'  Fastest teacher: {min(teacher_times):.1f}s')
        print(f'  Slowest teacher: {max(teacher_times):.1f}s')
    print(f'\nOutput directory: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()

