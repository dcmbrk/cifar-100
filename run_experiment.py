"""
Main Experiment Script
Chạy thực nghiệm đầy đủ theo báo cáo.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from torchvision import datasets, transforms

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from feature_extractor import load_or_extract_features
from similarity import compute_similarity_matrix, compute_similarity_matrix_memmap, load_similarity_memmap
from greedy_facility_location import greedy_facility_location
from baselines import BaselineSelector
from evaluation import CoresetEvaluator
from trainer import train_on_coreset
from visualization import (
    plot_class_distribution,
    plot_method_comparison,
    plot_budget_analysis,
    plot_greedy_progress,
    plot_training_curves,
    create_summary_figure
)


def setup_directories():
    """Tạo các thư mục cần thiết."""
    dirs = ['data', 'checkpoints', 'results', 'results/figures']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_cifar100(data_dir='./data'):
    """Load CIFAR-100 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def run_selection_methods(features, labels, similarity_matrix, k, seed=42, verbose=True):
    """
    Chạy tất cả các phương pháp selection.

    Args:
        features: Feature vectors
        labels: Labels
        similarity_matrix: Ma trận similarity
        k: Budget
        seed: Random seed
        verbose: In tiến trình

    Returns:
        results: Dict {method: (selected_indices, runtime, [scores])}
    """
    results = {}

    # Baseline selector
    baseline = BaselineSelector(features, labels, similarity_matrix)

    # 1. Random
    if verbose:
        print("\n[1/5] Running Random Selection...")
    selected, runtime = baseline.select('random', k, seed=seed)
    results['Random'] = {'indices': selected, 'runtime': runtime}

    # 2. Stratified Random
    if verbose:
        print("[2/5] Running Stratified Random Selection...")
    selected, runtime = baseline.select('stratified', k, seed=seed)
    results['Stratified'] = {'indices': selected, 'runtime': runtime}

    # 3. K-Center Greedy
    if verbose:
        print("[3/5] Running K-Center Greedy...")
    selected, runtime = baseline.select('k_center', k, verbose=verbose)
    results['K-Center'] = {'indices': selected, 'runtime': runtime}

    # 4. Herding
    if verbose:
        print("[4/5] Running Herding...")
    selected, runtime = baseline.select('herding', k, verbose=verbose)
    results['Herding'] = {'indices': selected, 'runtime': runtime}

    # 5. Greedy Facility Location
    if verbose:
        print("[5/5] Running Greedy Facility Location (Lazy)...")
    selected, scores, runtime = greedy_facility_location(
        similarity_matrix, k, lazy=True, verbose=verbose
    )
    results['Greedy FL'] = {'indices': selected, 'runtime': runtime, 'scores': scores}

    return results


def run_full_experiment(args):
    """Chạy thực nghiệm đầy đủ."""

    print("="*60)
    print("IMAGE SUMMARIZATION EXPERIMENT - CIFAR-100")
    print("="*60)
    print(f"Budget k: {args.budget}")
    print(f"Training epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("="*60)

    setup_directories()

    # Load data
    print("\n[Step 1] Loading CIFAR-100 dataset...")
    train_dataset, test_dataset = load_cifar100(args.data_dir)
    print(f"Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")

    # Extract features
    print("\n[Step 2] Extracting features with ResNet-18...")
    cache_path = Path(args.data_dir) / 'cifar100_features.npz'
    features, labels = load_or_extract_features(
        train_dataset,
        cache_path=str(cache_path),
        device=args.device,
        batch_size=args.batch_size
    )
    print(f"Features shape: {features.shape}")

    # Compute similarity matrix (dùng memmap để tiết kiệm RAM)
    print("\n[Step 3] Computing similarity matrix...")
    sim_cache = Path(args.data_dir) / 'cifar100_similarity.dat'
    n = features.shape[0]
    if sim_cache.exists():
        print(f"Loading from cache: {sim_cache}")
        similarity_matrix = load_similarity_memmap(str(sim_cache), n)
    else:
        similarity_matrix = compute_similarity_matrix_memmap(
            features, str(sim_cache), metric='cosine', batch_size=500
        )
        print(f"Saved to cache: {sim_cache}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Run selection methods
    print("\n[Step 4] Running selection methods...")
    selection_results = run_selection_methods(
        features, labels, similarity_matrix,
        k=args.budget,
        seed=args.seed,
        verbose=True
    )

    # Evaluate
    print("\n[Step 5] Evaluating selection methods...")
    evaluator = CoresetEvaluator(features, labels, similarity_matrix)

    all_metrics = {}
    for method, result in selection_results.items():
        print(f"\n--- {method} ---")
        metrics = evaluator.evaluate(result['indices'], verbose=True)
        metrics['runtime'] = result['runtime']
        all_metrics[method] = metrics

    # Comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    evaluator.compare_methods({m: r['indices'] for m, r in selection_results.items()})

    # Training (optional)
    if args.train:
        print("\n[Step 6] Training classifiers on coresets...")
        train_results = {}

        for method, result in selection_results.items():
            print(f"\n--- Training on {method} coreset ---")
            train_res = train_on_coreset(
                train_dataset,
                test_dataset,
                result['indices'],
                epochs=args.epochs,
                batch_size=64,
                lr=args.lr,
                device=args.device,
                verbose=True
            )
            train_results[method] = train_res
            all_metrics[method]['test_acc'] = train_res['test_acc']
            all_metrics[method]['train_time'] = train_res['train_time']

    # Visualization
    print("\n[Step 7] Generating visualizations...")
    results_dir = Path('results/figures')

    # Class distribution cho Greedy FL
    plot_class_distribution(
        labels,
        selection_results['Greedy FL']['indices'],
        save_path=results_dir / f'class_dist_greedy_k{args.budget}.png',
        title=f'Greedy FL Class Distribution (k={args.budget})'
    )

    # Method comparison
    plot_method_comparison(
        all_metrics,
        metrics=['fl_score', 'class_coverage', 'diversity'],
        save_path=results_dir / f'method_comparison_k{args.budget}.png'
    )

    # Greedy progress
    if 'scores' in selection_results['Greedy FL']:
        plot_greedy_progress(
            selection_results['Greedy FL']['scores'],
            save_path=results_dir / f'greedy_progress_k{args.budget}.png'
        )

    # Save results to JSON
    results_file = Path('results') / f'experiment_k{args.budget}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    # Convert numpy types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    save_data = {
        'config': vars(args),
        'metrics': convert_to_serializable(all_metrics),
        'selected_indices': {m: r['indices'] for m, r in selection_results.items()}
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED!")
    print("="*60)

    return all_metrics, selection_results


def run_budget_analysis(args):
    """Phân tích ảnh hưởng của budget k."""

    print("="*60)
    print("BUDGET ANALYSIS - CIFAR-100")
    print("="*60)

    setup_directories()

    # Load data
    print("\n[Step 1] Loading data and features...")
    train_dataset, test_dataset = load_cifar100(args.data_dir)

    cache_path = Path(args.data_dir) / 'cifar100_features.npz'
    features, labels = load_or_extract_features(
        train_dataset,
        cache_path=str(cache_path),
        device=args.device
    )

    sim_cache = Path(args.data_dir) / 'cifar100_similarity.dat'
    n = features.shape[0]
    if sim_cache.exists():
        similarity_matrix = load_similarity_memmap(str(sim_cache), n)
    else:
        similarity_matrix = compute_similarity_matrix_memmap(
            features, str(sim_cache), metric='cosine', batch_size=500
        )

    evaluator = CoresetEvaluator(features, labels, similarity_matrix)

    # Budgets to test
    budgets = [500, 1000, 2000, 5000]
    budget_results = {}

    print("\n[Step 2] Running Greedy FL for different budgets...")

    for k in budgets:
        print(f"\n--- Budget k = {k} ({100*k/len(labels):.1f}% of data) ---")

        selected, scores, runtime = greedy_facility_location(
            similarity_matrix, k, lazy=True, verbose=True
        )

        metrics = evaluator.evaluate(selected, verbose=False)
        metrics['runtime'] = runtime

        if args.train:
            print(f"Training classifier...")
            train_res = train_on_coreset(
                train_dataset, test_dataset, selected,
                epochs=args.epochs, device=args.device, verbose=False
            )
            metrics['test_acc'] = train_res['test_acc']

        budget_results[k] = metrics
        print(f"FL Score: {metrics['fl_score']:.4f}, "
              f"Coverage: {metrics['class_coverage']*100:.1f}%, "
              f"Time: {runtime:.1f}s")
        if 'test_acc' in metrics:
            print(f"Test Accuracy: {metrics['test_acc']:.2f}%")

    # Visualization
    print("\n[Step 3] Generating visualizations...")
    plot_budget_analysis(
        budget_results,
        save_path=Path('results/figures') / 'budget_analysis.png'
    )

    # Print summary table
    print("\n" + "="*60)
    print("BUDGET ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'k':>6} {'% Data':>8} {'FL Score':>10} {'Coverage':>10} {'Time (s)':>10}", end='')
    if args.train:
        print(f" {'Accuracy':>10}")
    else:
        print()
    print("-"*60)

    for k, metrics in budget_results.items():
        print(f"{k:>6} {100*k/len(labels):>7.1f}% {metrics['fl_score']:>10.4f} "
              f"{metrics['class_coverage']*100:>9.1f}% {metrics['runtime']:>10.1f}", end='')
        if 'test_acc' in metrics:
            print(f" {metrics['test_acc']:>9.2f}%")
        else:
            print()

    print("="*60)

    return budget_results


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 Image Summarization Experiment')

    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'budget', 'select_only'],
                        help='Experiment mode')
    parser.add_argument('--budget', type=int, default=1000,
                        help='Number of images to select (default: 1000)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for feature extraction')
    parser.add_argument('--train', action='store_true',
                        help='Train classifiers on coresets')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')

    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.mode == 'full':
        run_full_experiment(args)
    elif args.mode == 'budget':
        run_budget_analysis(args)
    elif args.mode == 'select_only':
        # Quick mode: chỉ chạy selection, không train
        args.train = False
        run_full_experiment(args)


if __name__ == '__main__':
    main()
