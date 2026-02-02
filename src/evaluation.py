"""
Evaluation Module
Đánh giá chất lượng tập ảnh được chọn theo các tiêu chí trong báo cáo.
"""

import numpy as np
from collections import Counter


def facility_location_score(similarity_matrix, selected_indices):
    """
    Tính Facility Location Score: f(S) = sum_{v in V} max_{s in S} sim(v, s)

    Đo mức độ "coverage" của tập S đối với toàn bộ V.

    Args:
        similarity_matrix: numpy array (n, n)
        selected_indices: List các indices được chọn

    Returns:
        score: FL Score (normalized by n)
    """
    if len(selected_indices) == 0:
        return 0.0

    selected_indices = list(selected_indices)
    max_sims = np.max(similarity_matrix[:, selected_indices], axis=1)

    # Normalize by n để có giá trị trong [0, 1]
    score = np.mean(max_sims)
    return score


def class_coverage(labels, selected_indices):
    """
    Tính Class Coverage: Tỷ lệ classes có ít nhất 1 ảnh trong S.

    Args:
        labels: numpy array (n,) - nhãn của tất cả ảnh
        selected_indices: List các indices được chọn

    Returns:
        coverage: Tỷ lệ classes được bao phủ (0-1)
        covered_classes: Số classes được bao phủ
        total_classes: Tổng số classes
    """
    total_classes = len(np.unique(labels))
    selected_labels = labels[selected_indices]
    covered_classes = len(np.unique(selected_labels))

    coverage = covered_classes / total_classes
    return coverage, covered_classes, total_classes


def class_distribution(labels, selected_indices):
    """
    Phân tích phân phối các classes trong tập được chọn.

    Args:
        labels: numpy array (n,)
        selected_indices: List các indices

    Returns:
        distribution: Dict {class_id: count}
        stats: Dict với min, max, mean, std
    """
    selected_labels = labels[selected_indices]
    distribution = dict(Counter(selected_labels))

    counts = list(distribution.values())
    stats = {
        'min': min(counts),
        'max': max(counts),
        'mean': np.mean(counts),
        'std': np.std(counts)
    }

    return distribution, stats


def diversity_score(features, selected_indices):
    """
    Tính Diversity Score: Trung bình khoảng cách giữa các ảnh trong S.

    Diversity cao = các ảnh được chọn đa dạng, không giống nhau.

    Args:
        features: numpy array (n, d) - feature vectors đã chuẩn hóa L2
        selected_indices: List các indices

    Returns:
        diversity: Average pairwise distance trong S
    """
    if len(selected_indices) < 2:
        return 0.0

    selected_features = features[selected_indices]
    k = len(selected_indices)

    # Tính pairwise distances
    # Với features đã chuẩn hóa L2: ||a-b||² = 2 - 2*cos(a,b) = 2 - 2*a.b
    similarities = np.dot(selected_features, selected_features.T)
    distances = np.sqrt(np.maximum(0, 2 - 2 * similarities))

    # Lấy upper triangle (không tính diagonal và phần đối xứng)
    triu_indices = np.triu_indices(k, k=1)
    pairwise_distances = distances[triu_indices]

    diversity = np.mean(pairwise_distances)
    return diversity


def redundancy_score(similarity_matrix, selected_indices):
    """
    Tính Redundancy Score: Đo mức độ trùng lặp trong tập S.

    Redundancy = trung bình max similarity của mỗi ảnh trong S với các ảnh khác trong S.
    Redundancy thấp = ít trùng lặp = tốt.

    Args:
        similarity_matrix: numpy array (n, n)
        selected_indices: List các indices

    Returns:
        redundancy: Average max similarity trong S (loại trừ chính nó)
    """
    if len(selected_indices) < 2:
        return 0.0

    selected_indices = list(selected_indices)
    k = len(selected_indices)

    # Ma trận similarity trong tập S
    sub_sim = similarity_matrix[np.ix_(selected_indices, selected_indices)]

    # Đặt diagonal = -inf để loại trừ chính nó
    np.fill_diagonal(sub_sim, -np.inf)

    # Max similarity của mỗi ảnh với các ảnh khác
    max_sims = np.max(sub_sim, axis=1)

    redundancy = np.mean(max_sims)
    return redundancy


def representation_error(features, labels, selected_indices):
    """
    Tính Representation Error: Trung bình khoảng cách từ mỗi ảnh đến ảnh gần nhất trong S.

    Error thấp = mỗi ảnh đều có đại diện gần trong S.

    Args:
        features: numpy array (n, d)
        labels: numpy array (n,)
        selected_indices: List các indices

    Returns:
        error: Average representation error
        per_class_error: Dict error theo từng class
    """
    n = features.shape[0]
    selected_features = features[selected_indices]

    # Khoảng cách từ mỗi ảnh đến các ảnh trong S
    # distances[i, j] = khoảng cách từ ảnh i đến ảnh selected[j]
    similarities = np.dot(features, selected_features.T)
    distances = np.sqrt(np.maximum(0, 2 - 2 * similarities))

    # Min distance cho mỗi ảnh
    min_distances = np.min(distances, axis=1)

    error = np.mean(min_distances)

    # Per-class error
    per_class_error = {}
    for cls in np.unique(labels):
        class_indices = np.where(labels == cls)[0]
        per_class_error[cls] = np.mean(min_distances[class_indices])

    return error, per_class_error


class CoresetEvaluator:
    """Class tổng hợp để đánh giá coreset."""

    def __init__(self, features, labels, similarity_matrix=None):
        """
        Args:
            features: numpy array (n, d)
            labels: numpy array (n,)
            similarity_matrix: numpy array (n, n), optional
        """
        self.features = features
        self.labels = labels
        self.n = features.shape[0]
        self.num_classes = len(np.unique(labels))

        if similarity_matrix is not None:
            self.similarity_matrix = similarity_matrix
        else:
            # Tính cosine similarity (với features đã chuẩn hóa L2)
            self.similarity_matrix = np.dot(features, features.T)

    def evaluate(self, selected_indices, verbose=True):
        """
        Đánh giá toàn diện tập coreset.

        Args:
            selected_indices: List các indices được chọn
            verbose: In kết quả

        Returns:
            metrics: Dict chứa tất cả các metrics
        """
        selected_indices = list(selected_indices)
        k = len(selected_indices)

        # Tính các metrics
        fl_score = facility_location_score(self.similarity_matrix, selected_indices)
        coverage, covered, total = class_coverage(self.labels, selected_indices)
        diversity = diversity_score(self.features, selected_indices)
        redundancy = redundancy_score(self.similarity_matrix, selected_indices)
        rep_error, per_class_error = representation_error(
            self.features, self.labels, selected_indices
        )
        distribution, dist_stats = class_distribution(self.labels, selected_indices)

        metrics = {
            'coreset_size': k,
            'coreset_ratio': k / self.n,
            'fl_score': fl_score,
            'class_coverage': coverage,
            'covered_classes': covered,
            'total_classes': total,
            'diversity': diversity,
            'redundancy': redundancy,
            'representation_error': rep_error,
            'class_distribution_stats': dist_stats
        }

        if verbose:
            print(f"\n{'='*50}")
            print(f"CORESET EVALUATION RESULTS")
            print(f"{'='*50}")
            print(f"Coreset size: {k} / {self.n} ({100*k/self.n:.2f}%)")
            print(f"{'='*50}")
            print(f"Facility Location Score: {fl_score:.4f}")
            print(f"Class Coverage: {coverage*100:.1f}% ({covered}/{total} classes)")
            print(f"Diversity Score: {diversity:.4f}")
            print(f"Redundancy Score: {redundancy:.4f}")
            print(f"Representation Error: {rep_error:.4f}")
            print(f"{'='*50}")
            print(f"Class Distribution:")
            print(f"  Min per class: {dist_stats['min']}")
            print(f"  Max per class: {dist_stats['max']}")
            print(f"  Mean per class: {dist_stats['mean']:.2f}")
            print(f"  Std per class: {dist_stats['std']:.2f}")
            print(f"{'='*50}")

        return metrics

    def compare_methods(self, results_dict, verbose=True):
        """
        So sánh kết quả của nhiều phương pháp.

        Args:
            results_dict: Dict {method_name: selected_indices}
            verbose: In bảng so sánh

        Returns:
            comparison: Dict {method_name: metrics}
        """
        comparison = {}

        for method, indices in results_dict.items():
            metrics = self.evaluate(indices, verbose=False)
            comparison[method] = metrics

        if verbose:
            print(f"\n{'='*80}")
            print(f"{'METHOD COMPARISON':^80}")
            print(f"{'='*80}")
            print(f"{'Method':<15} {'Size':>6} {'FL Score':>10} {'Coverage':>10} "
                  f"{'Diversity':>10} {'Redundancy':>10}")
            print(f"{'-'*80}")

            for method, metrics in comparison.items():
                print(f"{method:<15} {metrics['coreset_size']:>6} "
                      f"{metrics['fl_score']:>10.4f} "
                      f"{metrics['class_coverage']*100:>9.1f}% "
                      f"{metrics['diversity']:>10.4f} "
                      f"{metrics['redundancy']:>10.4f}")

            print(f"{'='*80}")

        return comparison
