import os
import json
import csv
import argparse
import matplotlib.pyplot as plt


def load_results(runs_dir="./runs"):
    results = []

    for name in os.listdir(runs_dir):
        exp_dir = os.path.join(runs_dir, name)
        result_path = os.path.join(exp_dir, "result.json")

        if os.path.isdir(exp_dir) and os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)

    return results


def save_csv(results, save_path):
    if not results:
        print("没有找到结果文件。")
        return

    keys = [
        "model_name", "loss_name", "best_val_acc",
        "test_acc", "test_loss", "precision",
        "recall", "f1", "roc_auc", "ap"
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in results:
            writer.writerow([r.get(k, "") for k in keys])

    print(f"已保存汇总表: {save_path}")


def plot_bar(results, metric, save_path):
    names = [r["model_name"] for r in results]
    values = [r[metric] for r in results]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values)
    plt.ylabel(metric)
    plt.title(f"Model Comparison - {metric}")
    plt.ylim(0, min(1.0, max(values) + 0.1))

    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存图像: {save_path}")


def print_markdown_table(results):
    print("\n论文表格可参考：\n")
    print("| 方法 | Accuracy | Precision | Recall | F1-score | AUC | AP |")
    print("|------|----------|-----------|--------|----------|-----|----|")
    for r in results:
        print(f"| {r['model_name']} | {r['test_acc']:.4f} | {r.get('precision', 0):.4f} | "
              f"{r.get('recall', 0):.4f} | {r.get('f1', 0):.4f} | "
              f"{r['roc_auc']:.4f} | {r['ap']:.4f} |")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize model training results.")
    parser.add_argument("--runs-dir", default="./runs")
    parser.add_argument("--save-dir", default="./runs_summary")
    return parser.parse_args()


def main():
    args = parse_args()
    runs_dir = args.runs_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    results = load_results(runs_dir)
    if not results:
        print("未找到任何 result.json，请先完成训练。")
        return

    # 按 test_acc 排序，论文展示更直观
    results = sorted(results, key=lambda x: x["test_acc"], reverse=True)

    save_csv(results, os.path.join(save_dir, "summary.csv"))

    plot_bar(results, "test_acc", os.path.join(save_dir, "compare_acc.png"))
    plot_bar(results, "f1", os.path.join(save_dir, "compare_f1.png"))
    plot_bar(results, "roc_auc", os.path.join(save_dir, "compare_auc.png"))

    print_markdown_table(results)


if __name__ == "__main__":
    main()
