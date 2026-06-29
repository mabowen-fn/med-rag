"""
Generate evaluation charts and visualizations for documentation
Creates comparison charts from evaluation results
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))


def plot_basic_evaluation_comparison(results_path="data/evaluation_report.json"):
    """Generate charts for basic evaluation (no_rag vs naive_rag vs hybrid_rag)"""
    logger.info(f"Loading basic evaluation results from {results_path}...")
    
    if not Path(results_path).exists():
        logger.warning(f"File not found: {results_path}. Skipping basic evaluation charts.")
        return
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = data['summary']
    methods = list(summary.keys())
    
    # Extract metrics
    bleu_scores = [summary[m]['avg_bleu'] for m in methods]
    rouge_scores = [summary[m]['avg_rouge'] for m in methods]
    bert_scores = [summary[m]['avg_bert_score'] for m in methods]
    hallucination_rates = [summary[m]['hallucination_rate'] * 100 for m in methods]
    response_times = [summary[m]['avg_response_time'] for m in methods]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Basic Evaluation: No RAG vs Naive RAG vs Hybrid RAG', fontsize=16, fontweight='bold')
    
    # 1. Quality Metrics Comparison
    x = np.arange(len(methods))
    width = 0.25
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width, bleu_scores, width, label='BLEU', color='#1f77b4')
    bars2 = ax1.bar(x, rouge_scores, width, label='ROUGE', color='#ff7f0e')
    bars3 = ax1.bar(x + width, bert_scores, width, label='BERTScore', color='#2ca02c')
    
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Quality Metrics', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['No RAG', 'Naive RAG', 'Hybrid RAG'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 2. Hallucination Rate
    ax2 = axes[0, 1]
    colors = ['#d62728' if h > 50 else '#ff7f0e' if h > 30 else '#2ca02c' for h in hallucination_rates]
    bars = ax2.bar(methods, hallucination_rates, color=colors, edgecolor='black')
    ax2.set_ylabel('Hallucination Rate (%)', fontsize=11)
    ax2.set_title('Hallucination Rate (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(['No RAG', 'Naive RAG', 'Hybrid RAG'])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, rate in zip(bars, hallucination_rates):
        ax2.text(bar.get_x() + bar.get_width()/2., rate + 1,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Response Time
    ax3 = axes[1, 0]
    bars = ax3.bar(methods, response_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    ax3.set_ylabel('Response Time (seconds)', fontsize=11)
    ax3.set_title('Response Time (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(['No RAG', 'Naive RAG', 'Hybrid RAG'])
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time in zip(bars, response_times):
        ax3.text(bar.get_x() + bar.get_width()/2., time + 0.1,
                f'{time:.2f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Improvement Summary
    ax4 = axes[1, 1]
    hybrid_bleu = bleu_scores[2]
    no_rag_bleu = bleu_scores[0]
    hybrid_rouge = rouge_scores[2]
    no_rag_rouge = rouge_scores[0]
    hybrid_bert = bert_scores[2]
    no_rag_bert = bert_scores[0]
    halluc_reduction = hallucination_rates[0] - hallucination_rates[2]
    
    improvements = [
        (hybrid_bleu - no_rag_bleu) / no_rag_bleu * 100,
        (hybrid_rouge - no_rag_rouge) / no_rag_rouge * 100,
        (hybrid_bert - no_rag_bert) / no_rag_bert * 100,
        halluc_reduction
    ]
    
    metrics = ['BLEU', 'ROUGE', 'BERTScore', 'Halluc.\nReduction']
    colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
    
    bars = ax4.bar(metrics, improvements, color=colors, edgecolor='black')
    ax4.set_ylabel('Improvement (%)', fontsize=11)
    ax4.set_title('Hybrid RAG Improvement vs No RAG', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        ax4.text(bar.get_x() + bar.get_width()/2., imp + 1,
                f'{imp:+.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/evaluation_charts_basic.png', dpi=300, bbox_inches='tight')
    logger.info("✅ Saved: data/evaluation_charts_basic.png")
    plt.close()


def plot_sota_comparison(results_path="data/sota_evaluation_report.json"):
    """Generate charts for SOTA comparison (baseline vs SOTA)"""
    logger.info(f"Loading SOTA evaluation results from {results_path}...")
    
    if not Path(results_path).exists():
        logger.warning(f"File not found: {results_path}. Skipping SOTA comparison charts.")
        return
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    comparison = data['comparison']
    
    # Extract metrics
    metrics = ['bleu_score', 'rouge_score', 'bert_score', 'hallucination_rate']
    baseline_values = [comparison[m]['baseline'] for m in metrics]
    sota_values = [comparison[m]['sota'] for m in metrics]
    
    # For hallucination, lower is better, so we invert for visualization
    halluc_baseline = comparison['hallucination_rate']['baseline'] * 100
    halluc_sota = comparison['hallucination_rate']['sota'] * 100
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('SOTA RAG vs Baseline RAG Comparison', fontsize=16, fontweight='bold')
    
    # 1. Quality Metrics
    ax1 = axes[0]
    x = np.arange(len(metrics) - 1)  # Exclude hallucination
    width = 0.35
    
    baseline_quality = baseline_values[:-1]
    sota_quality = sota_values[:-1]
    
    bars1 = ax1.bar(x - width/2, baseline_quality, width, label='Baseline RAG', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, sota_quality, width, label='SOTA RAG', color='#d62728')
    
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Quality Metrics', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['BLEU', 'ROUGE', 'BERTScore'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Hallucination Rate
    ax2 = axes[1]
    methods = ['Baseline RAG', 'SOTA RAG']
    halluc_values = [halluc_baseline, halluc_sota]
    colors = ['#1f77b4', '#2ca02c']
    
    bars = ax2.bar(methods, halluc_values, color=colors, edgecolor='black')
    ax2.set_ylabel('Hallucination Rate (%)', fontsize=11)
    ax2.set_title('Hallucination Rate (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, rate in zip(bars, halluc_values):
        ax2.text(bar.get_x() + bar.get_width()/2., rate + 0.5,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    reduction = halluc_baseline - halluc_sota
    reduction_pct = (reduction / halluc_baseline * 100) if halluc_baseline > 0 else 0
    ax2.annotate(f'Reduction: {reduction:.1f}%\n({reduction_pct:.1f}%)',
                xy=(1, halluc_sota), xytext=(1.3, halluc_sota + 5),
                fontsize=10, fontweight='bold', color='#2ca02c',
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2))
    
    plt.tight_layout()
    plt.savefig('data/evaluation_charts_sota.png', dpi=300, bbox_inches='tight')
    logger.info("✅ Saved: data/evaluation_charts_sota.png")
    plt.close()


def plot_ragas_metrics(results_path="data/ragas_evaluation_results.json"):
    """Generate charts for RAGAS evaluation"""
    logger.info(f"Loading RAGAS evaluation results from {results_path}...")
    
    if not Path(results_path).exists():
        logger.warning(f"File not found: {results_path}. Skipping RAGAS charts.")
        return
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    improvements = data['improvements']
    
    # Extract metrics
    metrics = list(improvements.keys())
    baseline_values = [improvements[m]['baseline'] for m in metrics]
    sota_values = [improvements[m]['sota'] for m in metrics]
    improvement_pcts = [improvements[m]['improvement_pct'] for m in metrics]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('RAGAS Evaluation: Baseline vs SOTA', fontsize=16, fontweight='bold')
    
    # 1. Metric Comparison
    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, sota_values, width, label='SOTA', color='#d62728')
    
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('RAGAS Metrics Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    
    # Format metric names
    metric_labels = [m.replace('_', ' ').title() for m in metrics]
    ax1.set_xticklabels(metric_labels, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 2. Improvement Percentage
    ax2 = axes[1]
    colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvement_pcts]
    
    bars = ax2.bar(metrics, improvement_pcts, color=colors, edgecolor='black')
    ax2.set_ylabel('Improvement (%)', fontsize=11)
    ax2.set_title('SOTA Improvement over Baseline', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, imp in zip(bars, improvement_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2., imp + 0.5,
                f'{imp:+.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/evaluation_charts_ragas.png', dpi=300, bbox_inches='tight')
    logger.info("✅ Saved: data/evaluation_charts_ragas.png")
    plt.close()


def generate_summary_table():
    """Generate a comprehensive summary table"""
    logger.info("Generating summary table...")
    
    summary_data = {}
    
    # Load basic evaluation
    basic_path = Path("data/evaluation_report.json")
    if basic_path.exists():
        with open(basic_path, 'r', encoding='utf-8') as f:
            basic = json.load(f)
        summary_data['basic'] = basic['summary']
    
    # Load SOTA evaluation
    sota_path = Path("data/sota_evaluation_report.json")
    if sota_path.exists():
        with open(sota_path, 'r', encoding='utf-8') as f:
            sota = json.load(f)
        summary_data['sota'] = sota['comparison']
    
    # Load RAGAS evaluation
    ragas_path = Path("data/ragas_evaluation_results.json")
    if ragas_path.exists():
        with open(ragas_path, 'r', encoding='utf-8') as f:
            ragas = json.load(f)
        summary_data['ragas'] = ragas['improvements']
    
    # Generate markdown table
    md_content = "# Evaluation Results Summary\n\n"
    
    if 'basic' in summary_data:
        md_content += "## Basic Evaluation: No RAG vs Naive RAG vs Hybrid RAG\n\n"
        md_content += "| Method | BLEU | ROUGE | BERTScore | Hallucination Rate | Response Time |\n"
        md_content += "|--------|------|-------|-----------|-------------------|---------------|\n"
        
        for method in ['no_rag', 'naive_rag', 'hybrid_rag']:
            s = summary_data['basic'][method]
            md_content += f"| {method} | {s['avg_bleu']:.4f} | {s['avg_rouge']:.4f} | "
            md_content += f"{s['avg_bert_score']:.4f} | {s['hallucination_rate']*100:.1f}% | "
            md_content += f"{s['avg_response_time']:.2f}s |\n"
        
        md_content += "\n"
    
    if 'sota' in summary_data:
        md_content += "## SOTA Comparison: Baseline vs SOTA\n\n"
        md_content += "| Metric | Baseline | SOTA | Improvement |\n"
        md_content += "|--------|----------|------|-------------|\n"
        
        for metric in ['bleu_score', 'rouge_score', 'bert_score', 'hallucination_rate']:
            m = summary_data['sota'][metric]
            baseline = m['baseline']
            sota = m['sota']
            
            if metric == 'hallucination_rate':
                baseline_str = f"{baseline*100:.1f}%"
                sota_str = f"{sota*100:.1f}%"
                improvement = f"{m['improvement']:.1f}% reduction"
            else:
                baseline_str = f"{baseline:.4f}"
                sota_str = f"{sota:.4f}"
                improvement = f"{m['improvement']:+.1f}%"
            
            md_content += f"| {metric} | {baseline_str} | {sota_str} | {improvement} |\n"
        
        md_content += "\n"
    
    if 'ragas' in summary_data:
        md_content += "## RAGAS Evaluation\n\n"
        md_content += "| Metric | Baseline | SOTA | Improvement |\n"
        md_content += "|--------|----------|------|-------------|\n"
        
        for metric, values in summary_data['ragas'].items():
            md_content += f"| {metric} | {values['baseline']:.4f} | {values['sota']:.4f} | "
            md_content += f"{values['improvement_pct']:+.1f}% |\n"
        
        md_content += "\n"
    
    # Save summary
    with open('data/evaluation_summary.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info("✅ Saved: data/evaluation_summary.md")


def main():
    """Generate all evaluation charts"""
    logger.info("=" * 80)
    logger.info("GENERATING EVALUATION CHARTS")
    logger.info("=" * 80)
    
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        logger.error("matplotlib is not installed. Please install it: pip install matplotlib")
        return
    
    # Generate charts
    plot_basic_evaluation_comparison()
    plot_sota_comparison()
    plot_ragas_metrics()
    generate_summary_table()
    
    logger.info("\n" + "=" * 80)
    logger.info("CHART GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nGenerated files:")
    logger.info("  - data/evaluation_charts_basic.png")
    logger.info("  - data/evaluation_charts_sota.png")
    logger.info("  - data/evaluation_charts_ragas.png")
    logger.info("  - data/evaluation_summary.md")
    logger.info("\nThese files can be used in your documentation.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

