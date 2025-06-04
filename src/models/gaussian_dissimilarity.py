import os
import numpy as np
import scipy
import sklearn
import torch
from tqdm.auto import tqdm
import multiprocessing as mp
import pickle


def shortest_path_worker(todo_queue, output_queue, nbg, target_nodes):
    while True:
        index = todo_queue.get()

        if index == -1:
            output_queue.put((-1, None))
            break

        _, predecessors = scipy.sparse.csgraph.dijkstra(
            nbg, directed=False, indices=target_nodes[index], return_predecessors=True)
        predecessors[predecessors == -9999] = -1

        # Convert to torch tensor before sending
        predecessors_torch = torch.from_numpy(
            predecessors).to(dtype=torch.int32)
        output_queue.put((index, predecessors_torch))


def find_shortest_paths(pairwise_dissimilarity_matrix, target_nodes=None, n_neighbors=20, max_processes=10):
    nbrs_alg = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors, metric="precomputed", n_jobs=-1)
    nbrs = nbrs_alg.fit(pairwise_dissimilarity_matrix)
    nbg = sklearn.neighbors.kneighbors_graph(
        nbrs, n_neighbors, metric="precomputed", mode="distance")

    if target_nodes is None:
        target_nodes = torch.arange(nbg.shape[0], dtype=torch.int32)

    geodesic_predecessor_matrix = torch.zeros(
        (target_nodes.shape[0], nbg.shape[1]), dtype=torch.int32)

    with tqdm(total=len(target_nodes) * nbg.shape[0], desc="Computing Shortest Paths") as pbar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()

        for i in tqdm(range(len(target_nodes)), desc="Preparing Dijkstra Inputs"):
            todo_queue.put(i)

        process_count = min(max_processes, mp.cpu_count())

        for i in tqdm(range(process_count), desc="Starting Processes"):
            todo_queue.put(-1)
            p = mp.Process(target=shortest_path_worker, args=(
                todo_queue, output_queue, nbg, target_nodes))
            p.start()

        finished_processes = 0
        while finished_processes != process_count:
            i, p = output_queue.get()

            if i == -1:
                finished_processes = finished_processes + 1
            else:
                geodesic_predecessor_matrix[i, :] = p
                pbar.update(len(p))

    del nbg
    del nbrs
    del nbrs_alg

    return geodesic_predecessor_matrix


def path_hops_worker(todo_queue, output_queue, predecessor_matrix):
    while True:
        i = todo_queue.get()
        if i is None:
            output_queue.put(None)
            break

        hops = 0
        current = torch.arange(predecessor_matrix.shape[1], dtype=torch.int32)
        active = (current != -1)
        while torch.any(active):
            current[active] = predecessor_matrix[i, current[active]]
            active = (current != -1)
            hops = hops + 1

        output_queue.put(hops)


def find_path_with_most_hops(predecessor_matrix):
    most_hops = 0
    with tqdm(total=predecessor_matrix.shape[0], desc="Computing longest paths") as pbar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()
        for i in tqdm(range(predecessor_matrix.shape[0]), desc="Preparing tasks"):
            todo_queue.put(i)

        for i in tqdm(range(mp.cpu_count()), desc="Starting processes"):
            todo_queue.put(None)
            p = mp.Process(target=path_hops_worker, args=(
                todo_queue, output_queue, predecessor_matrix))
            p.start()

        finished_processes = 0
        while finished_processes != mp.cpu_count():
            hops = output_queue.get()

            if hops is None:
                finished_processes = finished_processes + 1
            else:
                if hops > most_hops:
                    most_hops = hops
                pbar.update(1)

    return most_hops


def contract_path(predecessors, dissimilarity_choices, metric_to_contract):
    contractable = torch.full(predecessors.shape, True)

    while contractable.sum() > 0:
        # Get choice of dissimilarity metric from current node to predecessor
        # and from predecessor to predecessor of predecessor.
        current_choice = dissimilarity_choices
        # Fix: Replace -1 with 0 for safe indexing
        safe_predecessors = predecessors.clone()
        safe_predecessors[safe_predecessors == -1] = 0
        predecessors_choice = torch.take_along_dim(
            dissimilarity_choices, safe_predecessors.long(), dim=1)

        # Check which path sections are contractable and perform contraction
        safe_predecessors_of_predecessors = predecessors.clone()
        safe_predecessors_of_predecessors[safe_predecessors_of_predecessors == -1] = 0
        predecessors_of_predecessors = torch.take_along_dim(
            predecessors, safe_predecessors_of_predecessors.long(), dim=1)

        contractable = torch.logical_and(
            current_choice == metric_to_contract,
            predecessors_choice == metric_to_contract
        )
        contractable = torch.logical_and(
            contractable,
            predecessors != -1
        )
        contractable = torch.logical_and(
            contractable,
            predecessors_of_predecessors != -1
        )

        print(f"{contractable.sum()} path sections remain to be contracted")
        predecessors[contractable] = predecessors_of_predecessors[contractable]


class GaussianDissimilarityModel:
    def __init__(self, metrics, enable_path_contraction=True):
        self.metrics = metrics

        self.datapoint_count = metrics[0].get_datapoint_count()
        for metric in metrics[1:]:
            assert (metric.get_datapoint_count() == self.datapoint_count)

        self.enable_path_contraction = enable_path_contraction

    def generate_short_paths(self, total_path_count=40000, realization_count=8, variance_scale=0.1):
        assert (total_path_count % realization_count == 0)
        paths_per_realization = total_path_count // realization_count

        self.predecessor_matrix = torch.zeros(
            (total_path_count, self.datapoint_count), dtype=torch.int32)
        self.target_nodes = torch.randint(
            high=self.datapoint_count, size=(total_path_count,), dtype=torch.int32)
        self.dissimilarity_matrix_choices = torch.zeros(
            (total_path_count, self.datapoint_count), dtype=torch.int8)

        # Does not return anything, but does the processing...
        # Rounds determines how many times realizations should be drawn randomly
        for realization_index in tqdm(range(realization_count)):
            first_path_index = realization_index * paths_per_realization
            last_path_index = (realization_index + 1) * paths_per_realization

            print("Generating dissimilarity realizations...")
            dissimilarity_metrics_count = len(self.metrics)
            realizations = torch.zeros(
                (self.datapoint_count, self.datapoint_count, dissimilarity_metrics_count), dtype=torch.float32)
            for i, metric in enumerate(tqdm(self.metrics)):
                metric.get_realization(realizations[:, :, i], variance_scale)

            # For every datapoint pair, select smallest dissimilarity realization
            print("Choosing smallest dissimilarity realization pair-wise...")
            dissimilarity_matrix_choice = torch.argmin(
                realizations, dim=-1, keepdim=True)
            pairwise_dissimilarity_matrix = torch.gather(
                realizations, 2, dissimilarity_matrix_choice).squeeze(-1)

            # Run shortest path algorithm
            # dissimilarity_matrix_choices stores which type of dissimilarity (velocity model, adp model, ...) was used to go from datapoint x along
            # the path towards the target datapoint to the next hop.
            # It has shape (total_path_count, self.datapoint_count), so the first axis determines the path we are on (and hence also the target datapoint) and the
            # second axis determines the datapoint (node) from which the current hop starts.
            print("Running shortest path algorithm...")
            current_target_nodes = self.target_nodes[first_path_index:last_path_index]
            predecessors = find_shortest_paths(
                pairwise_dissimilarity_matrix, current_target_nodes)

            assert (torch.all(torch.sum((predecessors == -1).int(), dim=1) == 1))

            self.predecessor_matrix[first_path_index:last_path_index] = predecessors
            # Use torch.arange for indexing
            idx = torch.arange(self.datapoint_count).unsqueeze(
                0).repeat(predecessors.shape[0], 1)
            self.dissimilarity_matrix_choices[first_path_index:
                                              last_path_index] = dissimilarity_matrix_choice[idx, predecessors][..., 0]

            del pairwise_dissimilarity_matrix
            del dissimilarity_matrix_choice
            del realizations

        # Optional step for faster training: Contract predecessor matrix
        # Some dissimilarity metrics may be "contractable", which means that path A->B->C and path A->C have the same
        # mean, variance dissimilarity if all hops "->" refer to the same dissimilarity.
        # In that case, we can shorten the path by replacing the predecessor of C (which is B) with A.
        # We can detect this from the predecessor matrix by checking if an entry has the same dissimilarity type as its predecessor.
        # This algorithm has log(N) complexity, where N is the length of the longest path for the same dissimilarity type.
        if self.enable_path_contraction:
            for metric_type, metric in enumerate(self.metrics):
                if metric.is_contractable():
                    contract_path(self.predecessor_matrix,
                                  self.dissimilarity_matrix_choices, metric_type)
        # Determine new longest path after contraction
        print("Determining longest short path...")
        self.longest_shortest_path = find_path_with_most_hops(
            self.predecessor_matrix)
        print(f"Longest short path has {self.longest_shortest_path} hops")

    def get_longest_shortest_path(self):
        return self.longest_shortest_path

    def save(self, filename="gdm_model_torch.pkl"):
        data_dir = os.path.join(
            os.getcwd(), "../../data/processed/preprocessed")
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, filename)
        print(f"Saving model to {path}")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename="gdm_model_torch.pkl"):
        data_dir = os.path.join(
            os.getcwd(), "../../data/processed/preprocessed")
        path = os.path.join(data_dir, filename)
        print(f"Loading model from {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_random_short_paths(self, path_count, hop_skip_limit=None):
        # returns (paths, path_hops, total_dissimilarity_means, total_dissimilarity_variances)
        # where paths is of shape (path_count, maximum path length) and all others are of shape path_count

        # Target and source indices to cached predecessor matrix
        # Source indices are also datapoint indices, but target indices must be translated to datapoint indices
        # using self.target_nodes[path_target_indices]
        path_target_indices = torch.randint(
            self.predecessor_matrix.shape[0], size=(path_count,), dtype=torch.int32)
        path_source_indices = torch.randint(
            self.predecessor_matrix.shape[1], size=(path_count,), dtype=torch.int32)

        # Prevent pairs where both indices refer to the same datapoint
        mask_same = path_source_indices == self.target_nodes[path_target_indices]
        path_source_indices[mask_same] = (
            path_source_indices[mask_same] + 1) % self.predecessor_matrix.shape[1]

        current = path_source_indices.clone()
        paths = torch.zeros(
            (len(current), self.longest_shortest_path), dtype=torch.int32)
        path_hops = torch.zeros(len(current), dtype=torch.int32)

        for i in range(self.longest_shortest_path):
            paths[:, i] = current
            # previous = current
            active = (current != self.target_nodes[path_target_indices])
            if torch.any(active):
                current[active] = self.predecessor_matrix[path_target_indices[active], current[active]]
                path_hops[torch.logical_and(
                    active, current == self.target_nodes[path_target_indices])] = i + 1

        # Compute mean total dissimilarity as well as uncertainty about it (variance) along paths
        # Use provided models to compute means / variances for individual dissimilarity types
        # Assume that dissimilarity models are independent, i.e., variances are just added up
        dissim_choice = torch.take_along_dim(
            self.dissimilarity_matrix_choices[path_target_indices], paths[:, :-1].long(), dim=1)

        # Assume that p(d) from different models are entirely uncorrelated
        total_dissimilarity_means = torch.zeros(len(paths))
        total_dissimilarity_variances = torch.zeros(len(paths))

        for metric_type, metric in enumerate(self.metrics):
            means, variances = metric.mean_variance_along_path(
                paths, dissim_choice == metric_type)
            total_dissimilarity_means += means
            total_dissimilarity_variances += variances

        # Subsample paths such that there are no more than subsampled_pathhops hops
        if hop_skip_limit is not None:
            for i in range(len(paths)):
                l = int(
                    min(max(total_dissimilarity_means[i] / hop_skip_limit[i], 1), path_hops[i]))
                idxs = torch.linspace(0, path_hops[i], l+1, dtype=torch.int32)
                paths[i, :l+1] = paths[i, idxs]
                paths[i, l+1:] = paths[i, -1]
                path_hops[i] = l
        # print("Assertion check:")
        assert torch.all(path_hops <= paths.shape[1] - 1), (
            f"Found path_hops > max index: max(path_hops)={path_hops.max()} vs paths.shape[1]-1={paths.shape[1]-1}"
        )

        return paths, path_hops, total_dissimilarity_means, total_dissimilarity_variances
