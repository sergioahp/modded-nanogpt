#!/usr/bin/env python3
"""
Analyze token counts for speedrun records
"""
from pathlib import Path
import pandas as pd
import re

RECORDS_DIR = Path("/home/user/modded-nanogpt/records")

# Batch size schedule from train_gpt.py Hyperparameters
BATCH_SIZES = [
    8 * 2048 * 8,   # 131,072 tokens
    16 * 2048 * 8,  # 262,144 tokens
    24 * 2048 * 8,  # 393,216 tokens
]

def get_latest_speedruns(n=15):
    """Get the n latest speed run directories"""
    speedrun_dirs = []

    for track_dir in RECORDS_DIR.iterdir():
        if track_dir.is_dir() and track_dir.name.startswith('track_'):
            for speedrun_dir in track_dir.iterdir():
                if speedrun_dir.is_dir() and speedrun_dir.name.startswith('20'):
                    speedrun_dirs.append(speedrun_dir)

    # Sort by directory name (which includes date)
    speedrun_dirs.sort(key=lambda x: x.name, reverse=True)
    return speedrun_dirs[:n]

def get_final_step_from_log(log_path):
    """Extract the final step count from a log file"""
    with open(log_path, 'r') as f:
        for line in f:
            # Look for final step line: step:1845/1845 val_loss:...
            match = re.search(r'step:(\d+)/(\d+)\s+val_loss:', line)
            if match and match.group(1) == match.group(2):
                return int(match.group(1))
    return None

def calculate_tokens(num_steps):
    """
    Calculate total tokens processed given number of training steps.

    Based on train_gpt.py Hyperparameters:
    - num_scheduled_iterations: int = 1805
    - Batch size schedule transitions based on training progress

    The batch size schedule transitions at specific fractions of num_scheduled_iterations.
    For simplicity, we'll use a standard approximation:
    - Steps 0-601: 131,072 tokens/step
    - Steps 602-1203: 262,144 tokens/step
    - Steps 1204+: 393,216 tokens/step

    This assumes transitions at 1/3 and 2/3 of 1805 steps.
    """
    transition_1 = 1805 // 3  # ~601
    transition_2 = 2 * 1805 // 3  # ~1203

    total_tokens = 0

    for step in range(num_steps):
        if step < transition_1:
            total_tokens += BATCH_SIZES[0]
        elif step < transition_2:
            total_tokens += BATCH_SIZES[1]
        else:
            total_tokens += BATCH_SIZES[2]

    return total_tokens

def main():
    print("Analyzing token counts for latest 15 speedruns...")
    print("="*80)

    latest_speedruns = get_latest_speedruns(15)

    results = []

    for speedrun_dir in latest_speedruns:
        print(f"\nProcessing: {speedrun_dir.name}")

        # Get all log files
        log_files = list(speedrun_dir.glob("*.txt"))
        log_files = [f for f in log_files if f.name != "README.txt"]

        if not log_files:
            print(f"  No log files found")
            continue

        speedrun_results = []

        for log_file in log_files:
            run_id = log_file.stem
            final_step = get_final_step_from_log(log_file)

            if final_step:
                total_tokens = calculate_tokens(final_step)
                speedrun_results.append({
                    'speedrun': speedrun_dir.name,
                    'run_id': run_id,
                    'final_step': final_step,
                    'total_tokens': total_tokens,
                    'total_tokens_M': total_tokens / 1e6,
                    'total_tokens_B': total_tokens / 1e9,
                })

        if speedrun_results:
            print(f"  Runs analyzed: {len(speedrun_results)}")
            avg_tokens = sum(r['total_tokens'] for r in speedrun_results) / len(speedrun_results)
            print(f"  Average tokens: {avg_tokens/1e6:.1f}M ({avg_tokens/1e9:.3f}B)")
            results.extend(speedrun_results)
        else:
            print(f"  No valid runs found")

    # Create DataFrame
    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal runs analyzed: {len(df)}")
    print(f"Total speedruns: {df['speedrun'].nunique()}")

    # Summary by speedrun
    print(f"\n{'='*80}")
    print("PER-SPEEDRUN TOKEN STATISTICS")
    print(f"{'='*80}")

    for speedrun in df['speedrun'].unique():
        speedrun_data = df[df['speedrun'] == speedrun]
        print(f"\n{speedrun}:")
        print(f"  Number of runs: {len(speedrun_data)}")
        print(f"  Steps: {speedrun_data['final_step'].iloc[0]:,}")
        print(f"  Tokens per run: {speedrun_data['total_tokens'].iloc[0]:,} ({speedrun_data['total_tokens_M'].iloc[0]:.1f}M)")
        if len(speedrun_data) > 1:
            total_all_runs = speedrun_data['total_tokens'].sum()
            print(f"  Total tokens (all {len(speedrun_data)} runs): {total_all_runs:,} ({total_all_runs/1e9:.2f}B)")

    # Save to CSV
    output_csv = "/home/user/modded-nanogpt/token_analysis.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Full data saved to: {output_csv}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
