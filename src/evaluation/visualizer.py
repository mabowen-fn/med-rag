"""Visualization dashboard for Medical RAG evaluation results"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from loguru import logger


class RAGVisualizer:
    """Generate comprehensive visualizations for RAG evaluation"""
    
    def __init__(self, report_path: str = "data/evaluation_report.json"):
        self.report_path = Path(report_path)
        self.data = self._load_report()
        
        # Set style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def _load_report(self):
        """Load evaluation report"""
        if not self.report_path.exists():
            raise FileNotFoundError(f"Report not found at {self.report_path}")
        
        with open(self.report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def plot_method_comparison(self, save_path: str = "data/method_comparison.png"):
        """Bar chart comparing all methods across metrics"""
        logger.info("Generating method comparison chart...")
        
        methods = list(self.data["results"].keys())
        metrics = ["bleu", "rouge_l", "bert_score", "hallucination_rate"]
        
        # Prepare data
        data = []
        for method in methods:
            for metric in metrics:
                value = self.data["results"][method]["metrics"][metric]
                data.append({
                    "Method": method.replace("_", " ").title(),
                    "Metric": metric.replace("_", " ").title(),
                    "Value": value
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            metric_data = df[df["Metric"] == metric.replace("_", " ").title()]
            
            bars = ax.bar(
                metric_data["Method"],
                metric_data["Value"],
                palette="viridis",
                edgecolor="black",
                linewidth=1.5
            )
            
            ax.set_title(metric.replace("_", " ").title(), fontsize=14, fontweight="bold")
            ax.set_ylabel("Score", fontsize=12)
            ax.set_ylim(0, max(metric_data["Value"]) * 1.2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight="bold"
                )
        
        plt.suptitle("RAG Method Comparison Across Metrics", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved method comparison to {save_path}")
        plt.close()
    
    def plot_response_time_comparison(self, save_path: str = "data/response_time_comparison.png"):
        """Compare response times across methods"""
        logger.info("Generating response time comparison...")
        
        methods = list(self.data["results"].keys())
        times = [self.data["results"][m]["metrics"]["avg_response_time"] for m in methods]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(
            [m.replace("_", " ").title() for m in methods],
            times,
            color=["#FF6B6B", "#4ECDC4", "#45B7D1"],
            edgecolor="black",
            linewidth=1.5
        )
        
        ax.set_title("Response Time Comparison", fontsize=16, fontweight="bold")
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_ylim(0, max(times) * 1.2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}s',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight="bold"
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved response time comparison to {save_path}")
        plt.close()
    
    def plot_hallucination_comparison(self, save_path: str = "data/hallucination_comparison.png"):
        """Hallucination rate comparison (most important metric)"""
        logger.info("Generating hallucination rate comparison...")
        
        methods = list(self.data["results"].keys())
        hallucination_rates = [
            self.data["results"][m]["metrics"]["hallucination_rate"]
            for m in methods
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color based on rate (red = high, green = low)
        colors = []
        for rate in hallucination_rates:
            if rate < 0.3:
                colors.append("#2ECC71")  # Green
            elif rate < 0.5:
                colors.append("#F39C12")  # Orange
            else:
                colors.append("#E74C3C")  # Red
        
        bars = ax.bar(
            [m.replace("_", " ").title() for m in methods],
            hallucination_rates,
            color=colors,
            edgecolor="black",
            linewidth=1.5
        )
        
        ax.set_title("Hallucination Rate Comparison\n(Lower is Better)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Hallucination Rate", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.3, color='green', linestyle='--', linewidth=2, label="Target (< 30%)")
        ax.legend(fontsize=11)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height*100:.1f}%',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight="bold"
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved hallucination comparison to {save_path}")
        plt.close()
    
    def plot_improvement_analysis(self, save_path: str = "data/improvement_analysis.png"):
        """Show improvements of Hybrid RAG vs No RAG"""
        logger.info("Generating improvement analysis...")
        
        no_rag = self.data["results"]["no_rag"]["metrics"]
        hybrid = self.data["results"]["hybrid_rag"]["metrics"]
        
        metrics = {
            "BLEU": (hybrid["bleu"] - no_rag["bleu"]) / no_rag["bleu"] * 100,
            "ROUGE-L": (hybrid["rouge_l"] - no_rag["rouge_l"]) / no_rag["rouge_l"] * 100,
            "BERT Score": (hybrid["bert_score"] - no_rag["bert_score"]) / no_rag["bert_score"] * 100,
            "Response Time": (hybrid["avg_response_time"] - no_rag["avg_response_time"]) / no_rag["avg_response_time"] * 100,
            "Hallucination": (hybrid["hallucination_rate"] - no_rag["hallucination_rate"]) / no_rag["hallucination_rate"] * 100,
        }
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = ["#2ECC71" if v < 0 else "#E74C3C" for v in metrics.values()]
        
        bars = ax.barh(
            list(metrics.keys()),
            list(metrics.values()),
            color=colors,
            edgecolor="black",
            linewidth=1.5
        )
        
        ax.set_title("Hybrid RAG vs No RAG: Performance Changes", fontsize=16, fontweight="bold")
        ax.set_xlabel("Change (%)", fontsize=12)
        ax.axvline(x=0, color='black', linewidth=1.5)
        
        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            label_x = width + (1 if width > 0 else -1)
            ax.text(
                label_x,
                bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%',
                ha='left' if width > 0 else 'right',
                va='center',
                fontsize=11,
                fontweight="bold"
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved improvement analysis to {save_path}")
        plt.close()
    
    def plot_detailed_metrics_heatmap(self, save_path: str = "data/metrics_heatmap.png"):
        """Heatmap of all metrics across methods"""
        logger.info("Generating metrics heatmap...")
        
        methods = list(self.data["results"].keys())
        metrics_list = ["bleu", "rouge_l", "bert_score", "hallucination_rate", "avg_response_time"]
        
        # Prepare data
        data_matrix = []
        for method in methods:
            row = [self.data["results"][method]["metrics"][m] for m in metrics_list]
            data_matrix.append(row)
        
        df = pd.DataFrame(
            data_matrix,
            index=[m.replace("_", " ").title() for m in methods],
            columns=[m.replace("_", " ").title() for m in metrics_list]
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            cbar_kws={'label': 'Score'},
            ax=ax,
            linewidths=1,
            linecolor='black'
        )
        
        ax.set_title("RAG Evaluation Metrics Heatmap", fontsize=16, fontweight="bold", pad=20)
        ax.set_ylabel("Method", fontsize=12)
        ax.set_xlabel("Metric", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved metrics heatmap to {save_path}")
        plt.close()
    
    def generate_all_visualizations(self, output_dir: str = "data"):
        """Generate all visualizations"""
        logger.info("Generating comprehensive visualization dashboard...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.plot_method_comparison(str(output_path / "method_comparison.png"))
        self.plot_response_time_comparison(str(output_path / "response_time_comparison.png"))
        self.plot_hallucination_comparison(str(output_path / "hallucination_comparison.png"))
        self.plot_improvement_analysis(str(output_path / "improvement_analysis.png"))
        self.plot_detailed_metrics_heatmap(str(output_path / "metrics_heatmap.png"))
        
        logger.info(f"✅ All visualizations saved to {output_dir}/")
        logger.info("Generated:")
        logger.info("  - method_comparison.png")
        logger.info("  - response_time_comparison.png")
        logger.info("  - hallucination_comparison.png")
        logger.info("  - improvement_analysis.png")
        logger.info("  - metrics_heatmap.png")


if __name__ == "__main__":
    visualizer = RAGVisualizer()
    visualizer.generate_all_visualizations()
