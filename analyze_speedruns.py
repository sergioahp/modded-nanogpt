#!/usr/bin/env python3
"""
Analyze the latest speed run records: create dataframe and plot step vs loss curves
"""
import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Find all speed run directories
RECORDS_DIR = Path("/home/user/modded-nanogpt/records")

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

def parse_log_file(log_path):
    """Parse a single log file to extract step vs loss data"""
    steps = []
    val_losses = []
    train_times = []

    with open(log_path, 'r') as f:
        for line in f:
            # Look for lines with val_loss
            if 'step:' in line and 'val_loss:' in line:
                # Parse: step:1845/1845 val_loss:3.2800 train_time:114369ms step_avg:61.99ms
                match = re.search(r'step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+train_time:(\d+)ms', line)
                if match:
                    step = int(match.group(1))
                    val_loss = float(match.group(3))
                    train_time_ms = int(match.group(4))

                    steps.append(step)
                    val_losses.append(val_loss)
                    train_times.append(train_time_ms)

    return steps, val_losses, train_times

def load_speedrun_data(speedrun_dir):
    """Load all runs from a speed run directory"""
    runs_data = []

    # Get all .txt files (excluding README)
    log_files = [f for f in speedrun_dir.glob("*.txt") if f.name != "README.txt"]

    for log_file in log_files:
        run_id = log_file.stem  # UUID without .txt
        steps, val_losses, train_times = parse_log_file(log_file)

        if steps:  # Only include if we found data
            for step, val_loss, train_time in zip(steps, val_losses, train_times):
                runs_data.append({
                    'speedrun': speedrun_dir.name,
                    'run_id': run_id,
                    'step': step,
                    'val_loss': val_loss,
                    'train_time_ms': train_time
                })

    return runs_data

def get_commit_info(speedrun_dir):
    """Try to extract commit info from README or return speedrun name"""
    readme_path = speedrun_dir / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            content = f.read()
            # Look for commit hash patterns
            match = re.search(r'[0-9a-f]{7,40}', content)
            if match:
                return match.group(0)[:7]
    return speedrun_dir.name

def main():
    print("Loading speed run data...")

    # Get 15 latest speed runs
    latest_speedruns = get_latest_speedruns(15)

    print(f"\nFound {len(latest_speedruns)} latest speed runs:")
    for i, sr in enumerate(latest_speedruns):
        print(f"  {i+1}. {sr.name}")

    # Load all data
    all_data = []
    for speedrun_dir in latest_speedruns:
        print(f"\nLoading {speedrun_dir.name}...")
        runs_data = load_speedrun_data(speedrun_dir)
        all_data.extend(runs_data)
        print(f"  Loaded {len([r for r in runs_data if r['step'] == max([x['step'] for x in runs_data])])} runs")

    # Create DataFrame
    df = pd.DataFrame(all_data)

    print(f"\n{'='*80}")
    print("DATAFRAME SUMMARY")
    print(f"{'='*80}")
    print(f"Total records: {len(df)}")
    print(f"Speed runs: {df['speedrun'].nunique()}")
    print(f"Total runs: {df.groupby('speedrun')['run_id'].nunique().sum()}")
    print(f"\nDataFrame head:")
    print(df.head(20))

    # Save to CSV
    output_csv = "/home/user/modded-nanogpt/speedrun_analysis.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nDataFrame saved to: {output_csv}")

    # Create plots
    print(f"\n{'='*80}")
    print("CREATING PLOTS")
    print(f"{'='*80}")

    # Get unique speedruns and assign colors
    speedruns = df['speedrun'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(speedruns)))

    # Plot 1: Full view
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each run
    for i, speedrun in enumerate(speedruns):
        speedrun_data = df[df['speedrun'] == speedrun]

        # Get all unique runs for this speedrun
        run_ids = speedrun_data['run_id'].unique()

        for j, run_id in enumerate(run_ids):
            run_data = speedrun_data[speedrun_data['run_id'] == run_id].sort_values('step')
            # Add label only to first run of each speedrun for legend
            label = speedrun if j == 0 else None
            ax.plot(run_data['step'], run_data['val_loss'],
                   color=colors[i], alpha=0.25, linewidth=0.8, label=label)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Step vs Loss Curves for 15 Latest Speed Runs\n(All individual runs shown)',
                 fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_plot = "/home/user/modded-nanogpt/speedrun_plot.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"Full plot saved to: {output_plot}")
    plt.close()

    # Plot 2: Zoomed to last 50% of steps
    fig, ax = plt.subplots(figsize=(14, 8))

    # Find the midpoint step (50% mark) using 75th percentile to ignore outlier long runs
    final_steps_per_speedrun = df.groupby('speedrun')['step'].max()
    typical_max_step = final_steps_per_speedrun.quantile(0.75)
    min_step_zoom = typical_max_step * 0.5

    # Plot each run (only last 50% of data)
    for i, speedrun in enumerate(speedruns):
        speedrun_data = df[df['speedrun'] == speedrun]

        # Get all unique runs for this speedrun
        run_ids = speedrun_data['run_id'].unique()

        for j, run_id in enumerate(run_ids):
            run_data = speedrun_data[speedrun_data['run_id'] == run_id].sort_values('step')
            run_data_zoom = run_data[run_data['step'] >= min_step_zoom]
            if len(run_data_zoom) > 0:
                # Add label only to first run of each speedrun for legend
                label = speedrun if j == 0 else None
                ax.plot(run_data_zoom['step'], run_data_zoom['val_loss'],
                       color=colors[i], alpha=0.25, linewidth=0.8, label=label)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Step vs Loss Curves - Last 50% of Training\n(All individual runs shown)',
                 fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    # Set x-axis limits to focus on typical runs (ignore outlier long runs)
    ax.set_xlim(min_step_zoom, typical_max_step * 1.1)

    plt.tight_layout()
    output_plot_zoom = "/home/user/modded-nanogpt/speedrun_plot_zoomed.png"
    plt.savefig(output_plot_zoom, dpi=150, bbox_inches='tight')
    print(f"Zoomed plot (last 50%) saved to: {output_plot_zoom}")
    plt.close()

    # Calculate variance statistics
    print(f"\n{'='*80}")
    print("RUN-TO-RUN VARIANCE ANALYSIS")
    print(f"{'='*80}")

    # For each speedrun, calculate variance at the final step
    variance_report = []

    for speedrun in speedruns:
        speedrun_data = df[df['speedrun'] == speedrun]

        # Get final step for this speedrun
        final_step = speedrun_data['step'].max()
        final_data = speedrun_data[speedrun_data['step'] == final_step]

        if len(final_data) > 1:
            mean_loss = final_data['val_loss'].mean()
            std_loss = final_data['val_loss'].std()

            # Calculate how much loss variance corresponds to in steps
            # Use the mean learning curve to estimate
            speedrun_mean_curve = speedrun_data.groupby('step')['val_loss'].mean().reset_index()

            # Find how many steps would give similar loss difference
            # Compare to a step that's ~10% earlier
            earlier_step = int(final_step * 0.9)
            earlier_loss = speedrun_mean_curve[speedrun_mean_curve['step'] <= earlier_step]['val_loss'].iloc[-1] if len(speedrun_mean_curve[speedrun_mean_curve['step'] <= earlier_step]) > 0 else mean_loss

            loss_per_step = (earlier_loss - mean_loss) / (final_step - earlier_step) if final_step > earlier_step else 0
            equivalent_steps = std_loss / loss_per_step if loss_per_step > 0 else 0
            percent_training = (equivalent_steps / final_step * 100) if final_step > 0 else 0

            variance_report.append({
                'speedrun': speedrun,
                'num_runs': len(final_data),
                'mean_final_loss': mean_loss,
                'std_final_loss': std_loss,
                'final_step': final_step,
                'equivalent_steps': equivalent_steps,
                'percent_training': percent_training
            })

    variance_df = pd.DataFrame(variance_report)

    print("\nPer-speedrun variance:")
    for _, row in variance_df.iterrows():
        print(f"\n{row['speedrun']}:")
        print(f"  Runs: {row['num_runs']}")
        print(f"  Final loss: {row['mean_final_loss']:.4f} ± {row['std_final_loss']:.4f}")
        print(f"  Loss std ≈ {row['equivalent_steps']:.0f} steps ({row['percent_training']:.1f}% of training)")

    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL VARIANCE STATISTICS")
    print(f"{'='*80}")
    avg_std = variance_df['std_final_loss'].mean()
    avg_steps = variance_df['equivalent_steps'].mean()
    avg_percent = variance_df['percent_training'].mean()

    print(f"\nOn average across all {len(speedruns)} speed runs:")
    print(f"  The runs varied {avg_std:.4f} in final validation loss")
    print(f"  which is approximately {avg_steps:.0f} steps")
    print(f"  representing {avg_percent:.1f}% of total training steps")

    # Save variance report
    variance_csv = "/home/user/modded-nanogpt/variance_report.csv"
    variance_df.to_csv(variance_csv, index=False)
    print(f"\nVariance report saved to: {variance_csv}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
