"""
Visualization Module
Trực quan hóa kết quả thực nghiệm.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def plot_selected_images(dataset, selected_indices, labels, n_show=100,
                         n_cols=10, figsize=(15, 15), save_path=None, title=None):
    """
    Hiển thị các ảnh được chọn.

    Args:
        dataset: PyTorch dataset
        selected_indices: List indices được chọn
        labels: numpy array labels
        n_show: Số ảnh hiển thị
        n_cols: Số cột
        figsize: Kích thước figure
        save_path: Đường dẫn lưu file
        title: Tiêu đề
    """
    n_show = min(n_show, len(selected_indices))
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices[:n_show]):
        img, label = dataset[idx]
        if hasattr(img, 'numpy'):
            img = img.numpy()
        if img.shape[0] == 3:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        # Denormalize nếu cần
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f'Class {label}', fontsize=8)
        axes[i].axis('off')

    # Ẩn các axes thừa
    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
    return fig


def plot_tsne_projection(features, selected_indices_dict, figsize=(12, 10),
                         save_path=None, title=None, max_points=5000):
    """
    Vẽ t-SNE projection của toàn bộ dataset và các điểm được chọn.

    Args:
        features: numpy array (n, 512)
        selected_indices_dict: Dict {method_name: indices}
        figsize: Kích thước figure
        save_path: Đường dẫn lưu
        title: Tiêu đề
        max_points: Số điểm tối đa từ dataset gốc để vẽ (tránh lag)
    """
    from sklearn.manifold import TSNE
    import pandas as pd
    import seaborn as sns

    n = features.shape[0]
    if n > max_points:
        indices = np.random.choice(n, max_points, replace=False)
        features_subset = features[indices]
    else:
        features_subset = features
        indices = np.arange(n)

    print(f"Computing t-SNE for {len(features_subset)} points...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    embeds = tsne.fit_transform(features_subset)

    fig, ax = plt.subplots(figsize=figsize)

    # Vẽ toàn bộ dataset (mờ)
    ax.scatter(embeds[:, 0], embeds[:, 1], c='lightgrey', s=5, alpha=0.5, label='Dataset')

    # Ánh xạ selected_indices sang không gian subset nếu có
    # Hoặc vẽ trực tiếp nếu selected_indices nằm trong subset
    colors = plt.cm.get_cmap('tab10')

    for i, (method, selected) in enumerate(selected_indices_dict.items()):
        # Tìm các điểm selected nằm trong features_subset
        subset_mask = np.isin(indices, selected)
        if np.any(subset_mask):
            selected_embeds = embeds[subset_mask]
            ax.scatter(selected_embeds[:, 0], selected_embeds[:, 1],
                       s=30, label=method, alpha=0.8, edgecolors='white', linewidth=0.5)

    ax.set_title(title or "t-SNE Visualization of Selected Samples", fontsize=15)
    ax.legend(markerscale=2)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
    return fig


def plot_class_distribution(labels, selected_indices, class_names=None,
                            figsize=(14, 6), save_path=None, title=None):
    """
    Vẽ biểu đồ phân phối classes trong tập được chọn.

    Args:
        labels: numpy array (n,)
        selected_indices: List indices
        class_names: List tên các class (optional)
        figsize: Kích thước figure
        save_path: Đường dẫn lưu
        title: Tiêu đề
    """
    selected_labels = labels[selected_indices]
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    # Đếm số ảnh mỗi class
    counts = np.zeros(num_classes)
    for cls in unique_classes:
        counts[cls] = np.sum(selected_labels == cls)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(range(num_classes), counts, color='steelblue', edgecolor='navy')

    # Đánh dấu các class không có ảnh
    for i, count in enumerate(counts):
        if count == 0:
            bars[i].set_color('red')

    ax.set_xlabel('Class ID', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Class Distribution in Selected Set (k={len(selected_indices)})', fontsize=14)

    # Thêm đường trung bình
    mean_count = len(selected_indices) / num_classes
    ax.axhline(y=mean_count, color='red', linestyle='--', label=f'Mean: {mean_count:.1f}')
    ax.legend()

    # Annotations
    covered = np.sum(counts > 0)
    ax.annotate(f'Coverage: {covered}/{num_classes} classes ({100*covered/num_classes:.1f}%)',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
    return fig


def plot_method_comparison(comparison_results, metrics=['fl_score', 'class_coverage', 'diversity'],
                          figsize=(12, 4), save_path=None):
    """
    Vẽ biểu đồ so sánh các phương pháp.

    Args:
        comparison_results: Dict {method: metrics_dict}
        metrics: List các metrics cần vẽ
        figsize: Kích thước
        save_path: Đường dẫn lưu
    """
    methods = list(comparison_results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for i, metric in enumerate(metrics):
        values = [comparison_results[m][metric] for m in methods]

        if metric == 'class_coverage':
            values = [v * 100 for v in values]  # Convert to percentage

        bars = axes[i].bar(methods, values, color=colors)
        axes[i].set_title(metric.replace('_', ' ').title(), fontsize=12)
        axes[i].set_ylabel('Score')

        # Rotate labels nếu nhiều methods
        if len(methods) > 4:
            axes[i].set_xticklabels(methods, rotation=45, ha='right')

        # Thêm giá trị lên bar
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[i].annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
    return fig


def plot_budget_analysis(budget_results, figsize=(14, 5), save_path=None):
    """
    Vẽ biểu đồ phân tích ảnh hưởng của budget k.

    Args:
        budget_results: Dict {k: metrics_dict} hoặc List của (k, metrics)
        figsize: Kích thước
        save_path: Đường dẫn lưu
    """
    if isinstance(budget_results, dict):
        budgets = sorted(budget_results.keys())
        results = [budget_results[k] for k in budgets]
    else:
        budgets = [r[0] for r in budget_results]
        results = [r[1] for r in budget_results]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # FL Score vs Budget
    fl_scores = [r['fl_score'] for r in results]
    axes[0].plot(budgets, fl_scores, 'o-', color='blue', linewidth=2, markersize=8)
    axes[0].set_xlabel('Budget k')
    axes[0].set_ylabel('FL Score')
    axes[0].set_title('Facility Location Score vs Budget')
    axes[0].grid(True, alpha=0.3)

    # Accuracy vs Budget (nếu có)
    if 'test_acc' in results[0]:
        accuracies = [r['test_acc'] for r in results]
        axes[1].plot(budgets, accuracies, 's-', color='green', linewidth=2, markersize=8)
        axes[1].set_xlabel('Budget k')
        axes[1].set_ylabel('Test Accuracy (%)')
        axes[1].set_title('Downstream Accuracy vs Budget')
        axes[1].grid(True, alpha=0.3)
    else:
        # Diversity thay thế
        diversities = [r['diversity'] for r in results]
        axes[1].plot(budgets, diversities, 's-', color='orange', linewidth=2, markersize=8)
        axes[1].set_xlabel('Budget k')
        axes[1].set_ylabel('Diversity Score')
        axes[1].set_title('Diversity vs Budget')
        axes[1].grid(True, alpha=0.3)

    # Runtime vs Budget (nếu có)
    if 'runtime' in results[0]:
        runtimes = [r['runtime'] for r in results]
        axes[2].plot(budgets, runtimes, '^-', color='red', linewidth=2, markersize=8)
        axes[2].set_xlabel('Budget k')
        axes[2].set_ylabel('Runtime (seconds)')
        axes[2].set_title('Selection Time vs Budget')
        axes[2].grid(True, alpha=0.3)
    else:
        # Redundancy thay thế
        redundancies = [r['redundancy'] for r in results]
        axes[2].plot(budgets, redundancies, '^-', color='purple', linewidth=2, markersize=8)
        axes[2].set_xlabel('Budget k')
        axes[2].set_ylabel('Redundancy Score')
        axes[2].set_title('Redundancy vs Budget')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
    return fig


def plot_greedy_progress(scores, figsize=(10, 5), save_path=None):
    """
    Vẽ tiến trình của thuật toán Greedy (FL Score theo số iterations).

    Args:
        scores: List giá trị f(S) sau mỗi bước
        figsize: Kích thước
        save_path: Đường dẫn lưu
    """
    fig, ax = plt.subplots(figsize=figsize)

    iterations = range(1, len(scores) + 1)
    ax.plot(iterations, scores, '-', color='blue', linewidth=1.5)
    ax.fill_between(iterations, scores, alpha=0.3)

    ax.set_xlabel('Number of Selected Images', fontsize=12)
    ax.set_ylabel('Facility Location Score', fontsize=12)
    ax.set_title('Greedy Selection Progress', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Annotate điểm cuối
    ax.annotate(f'Final: {scores[-1]:.4f}',
                xy=(len(scores), scores[-1]),
                xytext=(-50, 20), textcoords='offset points',
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
    return fig


def plot_training_curves(history, figsize=(12, 4), save_path=None):
    """
    Vẽ training curves (loss và accuracy).

    Args:
        history: Dict với 'train_loss', 'train_acc', và optional 'val_loss', 'val_acc'
        figsize: Kích thước
        save_path: Đường dẫn lưu
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    if 'val_acc' in history:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
    return fig


def create_summary_figure(comparison_results, budget_results=None,
                          figsize=(16, 12), save_path=None):
    """
    Tạo figure tổng hợp kết quả thực nghiệm.

    Args:
        comparison_results: Dict {method: metrics}
        budget_results: Dict {k: metrics} (optional)
        figsize: Kích thước
        save_path: Đường dẫn lưu
    """
    fig = plt.figure(figsize=figsize)

    if budget_results:
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    else:
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    methods = list(comparison_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    # Row 1: Method comparison
    # FL Score
    ax1 = fig.add_subplot(gs[0, 0])
    fl_scores = [comparison_results[m]['fl_score'] for m in methods]
    bars = ax1.bar(methods, fl_scores, color=colors)
    ax1.set_title('Facility Location Score', fontsize=12)
    ax1.set_ylabel('Score')
    ax1.set_xticklabels(methods, rotation=45, ha='right')

    # Class Coverage
    ax2 = fig.add_subplot(gs[0, 1])
    coverages = [comparison_results[m]['class_coverage'] * 100 for m in methods]
    bars = ax2.bar(methods, coverages, color=colors)
    ax2.set_title('Class Coverage (%)', fontsize=12)
    ax2.set_ylabel('Coverage (%)')
    ax2.set_ylim(0, 105)
    ax2.set_xticklabels(methods, rotation=45, ha='right')

    # Diversity
    ax3 = fig.add_subplot(gs[0, 2])
    diversities = [comparison_results[m]['diversity'] for m in methods]
    bars = ax3.bar(methods, diversities, color=colors)
    ax3.set_title('Diversity Score', fontsize=12)
    ax3.set_ylabel('Score')
    ax3.set_xticklabels(methods, rotation=45, ha='right')

    # Row 2: Budget analysis (nếu có)
    if budget_results:
        budgets = sorted(budget_results.keys())
        results = [budget_results[k] for k in budgets]

        ax4 = fig.add_subplot(gs[1, 0])
        fl_by_budget = [r['fl_score'] for r in results]
        ax4.plot(budgets, fl_by_budget, 'o-', color='blue', linewidth=2, markersize=8)
        ax4.set_xlabel('Budget k')
        ax4.set_ylabel('FL Score')
        ax4.set_title('FL Score vs Budget')
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 1])
        div_by_budget = [r['diversity'] for r in results]
        ax5.plot(budgets, div_by_budget, 's-', color='green', linewidth=2, markersize=8)
        ax5.set_xlabel('Budget k')
        ax5.set_ylabel('Diversity')
        ax5.set_title('Diversity vs Budget')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 2])
        cov_by_budget = [r['class_coverage'] * 100 for r in results]
        ax6.plot(budgets, cov_by_budget, '^-', color='red', linewidth=2, markersize=8)
        ax6.set_xlabel('Budget k')
        ax6.set_ylabel('Coverage (%)')
        ax6.set_title('Class Coverage vs Budget')
        ax6.grid(True, alpha=0.3)

    fig.suptitle('Image Summarization Results - CIFAR-100', fontsize=16, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.close()
    return fig
