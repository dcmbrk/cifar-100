"""
Similarity Matrix Module
Tính toán ma trận tương đồng giữa các feature vectors.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from tqdm import tqdm
import os


def compute_similarity_matrix_memmap(features, output_path, metric='cosine', batch_size=500):
    """
    Tính similarity matrix và lưu vào file memmap (tiết kiệm RAM).

    Args:
        features: numpy array (n, d)
        output_path: Đường dẫn file .dat để lưu
        metric: 'cosine'
        batch_size: Batch size

    Returns:
        similarity_matrix: numpy memmap (n, n)
    """
    n = features.shape[0]

    # Tạo memmap file
    sim_matrix = np.memmap(output_path, dtype='float32', mode='w+', shape=(n, n))

    for i in tqdm(range(0, n, batch_size), desc="Computing similarity"):
        end_i = min(i + batch_size, n)
        sim_matrix[i:end_i] = np.dot(features[i:end_i], features.T).astype(np.float32)
        sim_matrix.flush()

    return sim_matrix


def load_similarity_memmap(path, n):
    """Load similarity matrix từ memmap file."""
    return np.memmap(path, dtype='float32', mode='r', shape=(n, n))


def compute_similarity_matrix(features, metric='cosine', gamma=None, batch_size=1000):
    """
    Tính ma trận tương đồng giữa các feature vectors.

    Args:
        features: numpy array shape (n, d) - các feature vectors đã chuẩn hóa L2
        metric: 'cosine' hoặc 'rbf'
        gamma: Tham số gamma cho RBF kernel (None = 1/d)
        batch_size: Batch size để tính toán (giảm memory usage)

    Returns:
        similarity_matrix: numpy array shape (n, n)
    """
    n = features.shape[0]

    if metric == 'cosine':
        # Với features đã chuẩn hóa L2, cosine similarity = dot product
        # Tính theo batch để tiết kiệm bộ nhớ
        if n <= batch_size:
            similarity_matrix = np.dot(features, features.T)
        else:
            similarity_matrix = np.zeros((n, n), dtype=np.float32)
            for i in tqdm(range(0, n, batch_size), desc="Computing similarity"):
                end_i = min(i + batch_size, n)
                similarity_matrix[i:end_i] = np.dot(features[i:end_i], features.T)

    elif metric == 'rbf':
        if gamma is None:
            gamma = 1.0 / features.shape[1]

        if n <= batch_size:
            similarity_matrix = rbf_kernel(features, gamma=gamma)
        else:
            similarity_matrix = np.zeros((n, n), dtype=np.float32)
            for i in tqdm(range(0, n, batch_size), desc="Computing similarity"):
                end_i = min(i + batch_size, n)
                similarity_matrix[i:end_i] = rbf_kernel(
                    features[i:end_i], features, gamma=gamma
                )
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'rbf'.")

    return similarity_matrix.astype(np.float32)


def compute_pairwise_distances(features, batch_size=1000):
    """
    Tính ma trận khoảng cách Euclidean giữa các feature vectors.

    Args:
        features: numpy array shape (n, d)
        batch_size: Batch size để tính toán

    Returns:
        distance_matrix: numpy array shape (n, n)
    """
    n = features.shape[0]

    if n <= batch_size:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        sq_norms = np.sum(features ** 2, axis=1)
        distances = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(features, features.T)
        distances = np.maximum(distances, 0)  # Tránh giá trị âm do lỗi số học
        distance_matrix = np.sqrt(distances)
    else:
        distance_matrix = np.zeros((n, n), dtype=np.float32)
        sq_norms = np.sum(features ** 2, axis=1)

        for i in tqdm(range(0, n, batch_size), desc="Computing distances"):
            end_i = min(i + batch_size, n)
            batch_features = features[i:end_i]

            distances = sq_norms[i:end_i, np.newaxis] + sq_norms[np.newaxis, :] \
                        - 2 * np.dot(batch_features, features.T)
            distances = np.maximum(distances, 0)
            distance_matrix[i:end_i] = np.sqrt(distances)

    return distance_matrix.astype(np.float32)


class SimilarityComputer:
    """Class wrapper để tính similarity theo yêu cầu (on-the-fly)."""

    def __init__(self, features, metric='cosine', precompute=True):
        """
        Args:
            features: numpy array shape (n, d)
            metric: 'cosine' hoặc 'rbf'
            precompute: Có tính trước toàn bộ ma trận hay không
        """
        self.features = features
        self.metric = metric
        self.n = features.shape[0]

        if precompute:
            self.similarity_matrix = compute_similarity_matrix(features, metric)
        else:
            self.similarity_matrix = None

    def get_similarity(self, i, j):
        """Lấy similarity giữa ảnh i và j."""
        if self.similarity_matrix is not None:
            return self.similarity_matrix[i, j]

        if self.metric == 'cosine':
            return np.dot(self.features[i], self.features[j])
        else:
            diff = self.features[i] - self.features[j]
            gamma = 1.0 / self.features.shape[1]
            return np.exp(-gamma * np.dot(diff, diff))

    def get_similarities_to_set(self, idx, selected_indices):
        """
        Lấy similarity từ ảnh idx đến tất cả ảnh trong selected_indices.

        Args:
            idx: Index của ảnh cần tính
            selected_indices: List các indices đã chọn

        Returns:
            similarities: numpy array shape (len(selected_indices),)
        """
        if len(selected_indices) == 0:
            return np.array([])

        if self.similarity_matrix is not None:
            return self.similarity_matrix[idx, selected_indices]

        selected_features = self.features[selected_indices]
        if self.metric == 'cosine':
            return np.dot(selected_features, self.features[idx])
        else:
            gamma = 1.0 / self.features.shape[1]
            diffs = selected_features - self.features[idx]
            sq_dists = np.sum(diffs ** 2, axis=1)
            return np.exp(-gamma * sq_dists)

    def get_max_similarity_to_set(self, selected_indices):
        """
        Với mỗi ảnh trong dataset, tính max similarity đến tập selected.

        Args:
            selected_indices: List các indices đã chọn

        Returns:
            max_sims: numpy array shape (n,) - max similarity của mỗi ảnh đến tập selected
        """
        if len(selected_indices) == 0:
            return np.zeros(self.n)

        if self.similarity_matrix is not None:
            return np.max(self.similarity_matrix[:, selected_indices], axis=1)

        # Tính on-the-fly
        selected_features = self.features[selected_indices]
        if self.metric == 'cosine':
            sims = np.dot(self.features, selected_features.T)
        else:
            gamma = 1.0 / self.features.shape[1]
            sims = rbf_kernel(self.features, selected_features, gamma=gamma)

        return np.max(sims, axis=1)
