"""
Baseline Methods
Các phương pháp baseline để so sánh với Greedy Facility Location.
"""

import numpy as np
from tqdm import tqdm
import time


def random_selection(n, k, seed=None):
    """
    Random Sampling - Chọn ngẫu nhiên k ảnh.

    Args:
        n: Tổng số ảnh
        k: Số lượng cần chọn
        seed: Random seed

    Returns:
        selected: List các indices được chọn
        runtime: Thời gian chạy (giây)
    """
    start_time = time.time()

    if seed is not None:
        np.random.seed(seed)

    selected = np.random.choice(n, size=k, replace=False).tolist()
    runtime = time.time() - start_time

    return selected, runtime


def stratified_random_selection(labels, k, seed=None):
    """
    Stratified Random Sampling - Chọn ngẫu nhiên k/num_classes ảnh từ mỗi lớp.

    Args:
        labels: numpy array (n,) - nhãn của từng ảnh
        k: Tổng số lượng cần chọn
        seed: Random seed

    Returns:
        selected: List các indices được chọn
        runtime: Thời gian chạy (giây)
    """
    start_time = time.time()

    if seed is not None:
        np.random.seed(seed)

    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    per_class = k // num_classes
    remainder = k % num_classes

    selected = []

    for i, cls in enumerate(unique_classes):
        class_indices = np.where(labels == cls)[0]
        n_select = per_class + (1 if i < remainder else 0)
        n_select = min(n_select, len(class_indices))

        chosen = np.random.choice(class_indices, size=n_select, replace=False)
        selected.extend(chosen.tolist())

    runtime = time.time() - start_time
    return selected, runtime


def k_center_greedy(features, k, verbose=True):
    """
    K-Center Greedy - Chọn điểm xa nhất với tập đã chọn.

    Thuật toán:
    1. Chọn điểm đầu tiên ngẫu nhiên
    2. Lặp: Chọn điểm có khoảng cách min đến tập S là lớn nhất

    Args:
        features: numpy array (n, d) - feature vectors
        k: Budget
        verbose: In tiến trình

    Returns:
        selected: List các indices được chọn
        runtime: Thời gian chạy (giây)
    """
    start_time = time.time()

    n = features.shape[0]
    selected = []

    # Chọn điểm đầu tiên ngẫu nhiên
    first_idx = np.random.randint(n)
    selected.append(first_idx)

    # Khoảng cách min từ mỗi điểm đến tập đã chọn
    min_distances = np.full(n, np.inf)

    # Cập nhật khoảng cách với điểm đầu tiên
    dists_to_first = np.linalg.norm(features - features[first_idx], axis=1)
    min_distances = np.minimum(min_distances, dists_to_first)

    iterator = tqdm(range(1, k), desc="K-Center Greedy") if verbose else range(1, k)

    for _ in iterator:
        # Chọn điểm có min_distance lớn nhất
        # Bỏ qua các điểm đã chọn
        masked_distances = min_distances.copy()
        masked_distances[selected] = -np.inf

        next_idx = np.argmax(masked_distances)
        selected.append(next_idx)

        # Cập nhật khoảng cách
        dists_to_new = np.linalg.norm(features - features[next_idx], axis=1)
        min_distances = np.minimum(min_distances, dists_to_new)

    runtime = time.time() - start_time
    return selected, runtime


def herding_selection(features, labels, k, verbose=True):
    """
    Herding - Chọn điểm gần centroid lớp nhất.

    Với mỗi lớp, chọn các điểm sao cho mean của các điểm đã chọn
    gần với mean của toàn bộ lớp.

    Args:
        features: numpy array (n, d) - feature vectors
        labels: numpy array (n,) - nhãn
        k: Tổng số cần chọn
        verbose: In tiến trình

    Returns:
        selected: List các indices được chọn
        runtime: Thời gian chạy (giây)
    """
    start_time = time.time()

    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    per_class = k // num_classes
    remainder = k % num_classes

    selected = []

    for i, cls in enumerate(unique_classes):
        class_indices = np.where(labels == cls)[0]
        class_features = features[class_indices]

        # Centroid của lớp
        class_mean = np.mean(class_features, axis=0)

        n_select = per_class + (1 if i < remainder else 0)
        n_select = min(n_select, len(class_indices))

        # Herding: chọn từng điểm để mean của tập đã chọn gần class_mean nhất
        class_selected = []
        remaining = set(range(len(class_indices)))

        for j in range(n_select):
            best_idx = None
            best_dist = np.inf

            current_sum = np.zeros_like(class_mean)
            if class_selected:
                current_sum = np.sum(class_features[class_selected], axis=0)

            for idx in remaining:
                new_sum = current_sum + class_features[idx]
                new_mean = new_sum / (j + 1)
                dist = np.linalg.norm(new_mean - class_mean)

                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            class_selected.append(best_idx)
            remaining.remove(best_idx)

        # Chuyển về global indices
        selected.extend([class_indices[idx] for idx in class_selected])

    runtime = time.time() - start_time
    return selected, runtime


class BaselineSelector:
    """Class wrapper cho các phương pháp baseline."""

    def __init__(self, features, labels, similarity_matrix=None):
        """
        Args:
            features: numpy array (n, d)
            labels: numpy array (n,)
            similarity_matrix: numpy array (n, n) - optional
        """
        self.features = features
        self.labels = labels
        self.similarity_matrix = similarity_matrix
        self.n = features.shape[0]

    def select(self, method, k, seed=None, verbose=True):
        """
        Chọn k samples theo phương pháp chỉ định.

        Args:
            method: 'random', 'stratified', 'k_center', 'herding'
            k: Budget
            seed: Random seed
            verbose: In tiến trình

        Returns:
            selected: List các indices
            runtime: Thời gian (giây)
        """
        if seed is not None:
            np.random.seed(seed)

        if method == 'random':
            return random_selection(self.n, k, seed)

        elif method == 'stratified':
            return stratified_random_selection(self.labels, k, seed)

        elif method == 'k_center':
            return k_center_greedy(self.features, k, verbose)

        elif method == 'herding':
            return herding_selection(self.features, self.labels, k, verbose)

        else:
            raise ValueError(f"Unknown method: {method}")
