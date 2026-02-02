"""
Greedy Facility Location Algorithm - Optimized Version
Thuật toán Greedy tối đa hóa hàm Facility Location với Lazy Evaluation.
Đã tối ưu hóa khởi tạo để tránh nghẽn I/O trên dữ liệu lớn.
"""

import numpy as np
import heapq
from tqdm import tqdm
import time


class FacilityLocation:
    def __init__(self, similarity_matrix):
        self.sim_matrix = similarity_matrix
        self.n = similarity_matrix.shape[0]

    def evaluate(self, selected_indices):
        if len(selected_indices) == 0:
            return 0.0
        # Tránh tính toán trên toàn bộ ma trận nếu không cần thiết
        selected_indices = list(selected_indices)
        # Sử dụng row access cho symmetric matrix để tối ưu memmap I/O
        max_sims = np.max(self.sim_matrix[selected_indices, :], axis=0)
        return np.sum(max_sims)

    def marginal_gain(self, current_max_sims, new_idx):
        """
        Đã tối ưu: Chỉ nhận current_max_sims để tránh tính toán lại bên trong.
        Sử dụng row access [new_idx, :] thay vì [:, new_idx] cho symmetric matrix.
        """
        new_sims = self.sim_matrix[new_idx, :]
        improvements = np.maximum(0, new_sims - current_max_sims)
        return np.sum(improvements)


class LazyGreedyFacilityLocation:
    def __init__(self, similarity_matrix, features=None, verbose=True):
        self.facility_loc = FacilityLocation(similarity_matrix)
        self.features = features
        self.n = similarity_matrix.shape[0] if similarity_matrix is not None else features.shape[0]
        self.verbose = verbose

    def select(self, k):
        start_time = time.time()

        selected = []
        selected_set = set()
        current_max_sims = np.zeros(self.n, dtype='float32')
        scores = []

        # --- TỐI ƯU HÓA BƯỚC KHỞI TẠO ---
        if self.verbose:
            print(f"Initializing: Calculating initial gains for {self.n} points...")

        if self.features is not None:
            # Tối ưu O(nd): g_0(v) = (sum_{u in V} u)^T v
            # Với features đã chuẩn hóa L2, similarity = dot product
            feature_sum = np.sum(self.features, axis=0)
            initial_gains = np.dot(self.features, feature_sum)
        else:
            # Thay vì loop 50,000 lần hoặc dùng sum(axis=0) trực tiếp trên memmap (có thể chậm),
            # ta tính theo batch để có progress bar và tận dụng sequential read.
            initial_gains = np.zeros(self.n, dtype='float32')
            batch_size = 1000
            for i in tqdm(range(0, self.n, batch_size), desc="Initial scan", disable=not self.verbose):
                end_i = min(i + batch_size, self.n)
                # sum(axis=1) là row sum, rất nhanh cho C-order memmap
                initial_gains[i:end_i] = self.facility_loc.sim_matrix[i:end_i, :].sum(axis=1)

        # Build heap trong O(N)
        # Priority queue: (-gain, iteration_computed, idx)
        heap = [(-initial_gains[i], 0, i) for i in range(self.n)]
        heapq.heapify(heap)

        if self.verbose:
            print("Initialization complete. Starting Lazy Greedy selection...")

        iteration = 0
        n_evaluations = self.n

        iterator = tqdm(range(k), desc="Lazy Greedy") if self.verbose else range(k)

        for step in iterator:
            iteration += 1
            best_idx = None

            while True:
                neg_gain, last_computed, idx = heapq.heappop(heap)

                if idx in selected_set:
                    continue

                # Nếu gain này vừa mới được cập nhật ở vòng lặp này, nó chắc chắn là tốt nhất (do tính submodular)
                if last_computed == iteration:
                    best_idx = idx
                    break
                else:
                    # Re-evaluate marginal gain
                    new_gain = self.facility_loc.marginal_gain(current_max_sims, idx)
                    n_evaluations += 1
                    heapq.heappush(heap, (-new_gain, iteration, idx))

            # Cập nhật trạng thái
            selected.append(best_idx)
            selected_set.add(best_idx)

            # Cập nhật current_max_sims: max(current_max, độ tương đồng với điểm mới chọn)
            # Sử dụng row access [best_idx, :]
            new_sims = self.facility_loc.sim_matrix[best_idx, :]
            current_max_sims = np.maximum(current_max_sims, new_sims)

            current_score = np.sum(current_max_sims)
            scores.append(current_score)

        runtime = time.time() - start_time

        if self.verbose:
            print(f"\nSelection finished in {runtime:.2f}s")
            print(f"Total gain evaluations: {n_evaluations} (vs {self.n * k} naive)")
            print(f"Speedup factor: {(self.n * k) / n_evaluations:.2f}x")

        return selected, scores, runtime


def greedy_facility_location(similarity_matrix, k, features=None, lazy=True, verbose=True):
    """
    Wrapper function.
    Mặc định sử dụng LazyGreedy vì nó hiệu quả hơn rất nhiều cho tập dữ liệu lớn.
    """
    # Ép buộc dùng Lazy nếu dữ liệu lớn (> 10,000)
    if similarity_matrix is not None and similarity_matrix.shape[0] > 10000:
        lazy = True
    elif features is not None and features.shape[0] > 10000:
        lazy = True

    if lazy:
        algorithm = LazyGreedyFacilityLocation(similarity_matrix, features, verbose)
    else:
        # Giữ lại class GreedyFacilityLocation cũ của bạn nếu cần,
        # nhưng LazyGreedy ở trên đã được tối ưu vượt trội.
        from .greedy_facility_location_base import GreedyFacilityLocation # Giả sử bạn tách file
        algorithm = GreedyFacilityLocation(similarity_matrix, verbose)

    return algorithm.select(k)