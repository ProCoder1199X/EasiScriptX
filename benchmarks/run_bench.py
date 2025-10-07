#!/usr/bin/env python3
"""
EasiScriptX Benchmark Visualization Script
Generates bar plots and markdown tables from benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_benchmark_data(csv_file="benchmark_results.csv"):
    """Load benchmark data from CSV file."""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run benchmark.cpp first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    return df

def create_bar_plot(df, output_file="benchmark_plot.png"):
    """Create bar plot comparing ESX vs baseline performance."""
    # Filter successful benchmarks
    df_success = df[df['Success'] == True]
    
    if df_success.empty:
        print("No successful benchmarks to plot.")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Execution Time Comparison
    x = np.arange(len(df_success))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df_success['ESX_Time_us'], width, label='EasiScriptX', color='#2E86AB')
    bars2 = ax1.bar(x + width/2, df_success['Baseline_Time_us'], width, label='Baseline', color='#A23B72')
    
    ax1.set_xlabel('Benchmark')
    ax1.set_ylabel('Execution Time (μs)')
    ax1.set_title('EasiScriptX vs Baseline Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_success['Benchmark'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Speedup
    colors = ['#F18F01' if speedup >= 1.0 else '#C73E1D' for speedup in df_success['Speedup']]
    bars3 = ax2.bar(x, df_success['Speedup'], color=colors)
    
    ax2.set_xlabel('Benchmark')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('EasiScriptX Speedup vs Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_success['Benchmark'], rotation=45, ha='right')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bar plot saved to: {output_file}")

def create_memory_plot(df, output_file="memory_plot.png"):
    """Create memory usage plot."""
    df_success = df[df['Success'] == True]
    
    if df_success.empty:
        print("No successful benchmarks to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_success))
    bars = ax.bar(x, df_success['Memory_MB'], color='#2E86AB')
    
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage by Benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(df_success['Benchmark'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Memory plot saved to: {output_file}")

def generate_markdown_table(df, output_file="benchmark_table.md"):
    """Generate markdown table from benchmark results."""
    df_success = df[df['Success'] == True]
    
    if df_success.empty:
        print("No successful benchmarks to include in table.")
        return
    
    # Create markdown table
    md_content = "# EasiScriptX Performance Benchmarks\n\n"
    md_content += "## Results Summary\n\n"
    
    # Calculate summary statistics
    avg_speedup = df_success['Speedup'].mean()
    max_speedup = df_success['Speedup'].max()
    min_speedup = df_success['Speedup'].min()
    total_memory = df_success['Memory_MB'].sum()
    
    md_content += f"- **Average Speedup**: {avg_speedup:.2f}x\n"
    md_content += f"- **Maximum Speedup**: {max_speedup:.2f}x\n"
    md_content += f"- **Minimum Speedup**: {min_speedup:.2f}x\n"
    md_content += f"- **Total Memory Usage**: {total_memory:.1f} MB\n"
    md_content += f"- **Successful Benchmarks**: {len(df_success)}/{len(df)}\n\n"
    
    # Performance targets
    md_content += "## Performance Targets\n\n"
    md_content += "- **Matrix Multiplication**: Target 3x speedup via Eigen optimization\n"
    md_content += "- **Memory Usage**: Target 50% reduction via memory optimization\n"
    md_content += "- **LoRA Operations**: Target 2x speedup via optimized low-rank operations\n"
    md_content += "- **Attention**: Target 1.5x speedup via FlashAttention-2\n\n"
    
    # Detailed results table
    md_content += "## Detailed Results\n\n"
    md_content += "| Benchmark | ESX Time (μs) | Baseline Time (μs) | Speedup | Memory (MB) | Status |\n"
    md_content += "|-----------|---------------|-------------------|---------|-------------|--------|\n"
    
    for _, row in df_success.iterrows():
        status = "✅ Pass" if row['Speedup'] >= 1.0 else "⚠️ Slower"
        md_content += f"| {row['Benchmark']} | {row['ESX_Time_us']:.1f} | {row['Baseline_Time_us']:.1f} | {row['Speedup']:.2f}x | {row['Memory_MB']:.1f} | {status} |\n"
    
    # Failed benchmarks
    df_failed = df[df['Success'] == False]
    if not df_failed.empty:
        md_content += "\n## Failed Benchmarks\n\n"
        md_content += "| Benchmark | Status |\n"
        md_content += "|-----------|--------|\n"
        for _, row in df_failed.iterrows():
            md_content += f"| {row['Benchmark']} | ❌ Failed |\n"
    
    # Performance analysis
    md_content += "\n## Performance Analysis\n\n"
    
    # Best performing benchmarks
    best_benchmarks = df_success.nlargest(3, 'Speedup')
    md_content += "### Top Performing Benchmarks\n\n"
    for _, row in best_benchmarks.iterrows():
        md_content += f"- **{row['Benchmark']}**: {row['Speedup']:.2f}x speedup\n"
    
    # Memory efficiency
    memory_efficient = df_success[df_success['Memory_MB'] < df_success['Memory_MB'].mean()]
    if not memory_efficient.empty:
        md_content += "\n### Memory Efficient Benchmarks\n\n"
        for _, row in memory_efficient.iterrows():
            md_content += f"- **{row['Benchmark']}**: {row['Memory_MB']:.1f} MB\n"
    
    # Recommendations
    md_content += "\n## Recommendations\n\n"
    if avg_speedup >= 2.0:
        md_content += "✅ **Excellent Performance**: EasiScriptX shows significant speedup over baseline implementations.\n\n"
    elif avg_speedup >= 1.5:
        md_content += "✅ **Good Performance**: EasiScriptX shows good speedup over baseline implementations.\n\n"
    elif avg_speedup >= 1.0:
        md_content += "⚠️ **Moderate Performance**: EasiScriptX shows modest speedup over baseline implementations.\n\n"
    else:
        md_content += "❌ **Performance Issues**: EasiScriptX is slower than baseline implementations. Optimization needed.\n\n"
    
    # Optimization suggestions
    slow_benchmarks = df_success[df_success['Speedup'] < 1.0]
    if not slow_benchmarks.empty:
        md_content += "### Optimization Opportunities\n\n"
        for _, row in slow_benchmarks.iterrows():
            md_content += f"- **{row['Benchmark']}**: Consider optimizing implementation for better performance\n"
    
    # Save markdown file
    with open(output_file, 'w') as f:
        f.write(md_content)
    
    print(f"Markdown table saved to: {output_file}")

def main():
    """Main function to run all visualizations."""
    print("EasiScriptX Benchmark Visualization")
    print("==================================")
    
    # Load data
    df = load_benchmark_data()
    print(f"Loaded {len(df)} benchmark results")
    
    # Create visualizations
    create_bar_plot(df)
    create_memory_plot(df)
    generate_markdown_table(df)
    
    print("\nVisualization complete!")
    print("Generated files:")
    print("- benchmark_plot.png: Performance comparison bar plot")
    print("- memory_plot.png: Memory usage plot")
    print("- benchmark_table.md: Markdown table with results")

if __name__ == "__main__":
    main()
