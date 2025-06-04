import tensorflow as tf
import torch
import abc
import math


class GaussianDissimilarityMetric(abc.ABC):
    """
    Class that models an uncertain dissimilarity metric, where each dissimilarity is normally distributed.
    """

    @abc.abstractmethod
    def get_realization(self, output_matrix, variance_scale):
        """
        Retrieve a relization of the dissimilarity matrix of shape `(datapoint_count, datapoint_count)`.
        Entries with value `torch.inf` mean that there is no known path according to this dissimilarity metric.
        The scale of the output must be consistent across all dissimilarity metrics.
        The convention is to provide the dissimilarities in meters.
        To speed up this processing step, the result is not provided as a return value,
        but is written to the pre-allocated buffer provided as a parameter.

        :param output_matrix: The realization of the dissimilarity matrix, NumPy array.
        :param variance_scale: Scaling factor for the variance. May want to scale down variance if number of realizations is small to achieve result close to mean.
        """
        pass

    @abc.abstractmethod
    def mean_variance_along_path(self, paths, mask):
        """
        Assuming the transmitter moves along the provided path sections of shape `(path count, maximum path length)`,
        but only along those sections where corresponding mask of same shape is True, calculate the mean and variance
        of the distribution of the summed up dissimilarities, which are assumed to be normally distributed.

        This function can model correlations between individual dissimilarities of the same type as desired.

        Different dissimilarity types are always assumed to be uncorrelated by the current model.

        The implementation with constant-hop count paths and mask allows for better vecorization as opposed to an approach
        where the paths are of variable hop count.

        :param paths: Datapoint indices of path sections, NumPy array of shape `(path count, maximum path length)`
        :param mask: Boolean NumPy Array that indicates whether path secction should be included in the sum or ignored
        :return:
            - mean_sum - Mean of sum of dissimilarities
            - variance_sum - Variance of sum of dissimilarities
        """
        pass

    @abc.abstractmethod
    def get_datapoint_count(self):
        """
        Query the number of datapoints for which this metric provides dissimilarities.
        """
        pass

    @abc.abstractmethod
    def is_contractable(self):
        """
        Query whether this type of dissimilarity metric can be contracted.
        Path contraction eliminates hops A->B->C where both hops take the same dissimilarity metric.
        This requires an implementation the returns the same mean / variance for paths A->B->C and A->C,
        which is only possible in special cases.
        """
        pass


##################################################
# Angle Delay Profile-Based Dissimilarity Metric #
##################################################
@torch.jit.script
def compute_adp_dissimilarity_matrix(csi_array):
    L = csi_array.shape[0]
    output = torch.zeros((L, L), dtype=torch.float32, device=csi_array.device)
    powers = torch.einsum("lbrmt,lbrmt->lbt", csi_array, csi_array.conj()).real

    for i in range(L):
        w = csi_array[i:]
        h = csi_array[i]
        dotproducts = torch.abs(torch.square(
            torch.einsum("brmt,lbrmt->lbt", h.conj(), w)))
        d_new = torch.sum(1 - dotproducts /
                          (powers[i] * powers[i:]).real, dim=(1, 2))
        d = torch.clamp(d_new, min=0)
        output[i, i:] = d  # Fill upper triangle row-by-row

    # If you want a symmetric matrix as in the TF code:
    dissim_upper_tri = output
    dissim_matrix = dissim_upper_tri + dissim_upper_tri.T

    return dissim_matrix


class ADPDissimilarityMetric(GaussianDissimilarityMetric):
    def __init__(self, csi_time_domain, adp_to_mean_variance_distance_func):
        print("Computing ADP dissimilarity matrix...")
        adp_dissimilarity_matrix = compute_adp_dissimilarity_matrix(
            csi_time_domain)
        print("ADP dissimilarity matrix computed")
        print("ADP dissimilarity matrix shape:",
              adp_dissimilarity_matrix.shape)
        self.adp_distance_mean, self.adp_distance_variance = adp_to_mean_variance_distance_func(
            adp_dissimilarity_matrix)

    def get_realization(self, output_matrix, variance_scale):
        finite_distances = torch.triu(self.adp_distance_mean != torch.inf)
        mean_vals = self.adp_distance_mean[finite_distances]
        std_vals = torch.sqrt(
            self.adp_distance_variance[finite_distances] * variance_scale)
        random_numbers = torch.abs(torch.normal(mean=mean_vals, std=std_vals))
        output_matrix.fill_(float('inf'))
        output_matrix[finite_distances] = random_numbers
        output_matrix.T[finite_distances] = random_numbers  # Make symmetric

    def mean_variance_along_path(self, paths, mask):
        # Assume that ADP dissimilarity observations are perfectly uncorrelated
        mean = self.adp_distance_mean[paths[:, :-1], paths[:, 1:]]
        variance = self.adp_distance_variance[paths[:, :-1], paths[:, 1:]]

        mean_sum = torch.sum(torch.where(mask, mean, 0), axis=1)
        variance_sum = torch.sum(torch.where(mask, variance, 0), axis=1)

        return mean_sum, variance_sum

    def get_datapoint_count(self):
        return self.adp_distance_mean.shape[0]

    def is_contractable(self):
        return False

    def estimate_velocity(self, timestamps):
        # TODO: Estimate true velocity from ADP dissimilarities
        pass

#############################################
# Velocity Model-Based Dissimilarity Metric #
#############################################


class SimpleGaussianProcess:
    """
    Models a stationary Gaussian stochastic process that is either perfectly uncorrelated
    or perfectly correlated.
    """

    def __init__(self, process_mean, process_variance, perfectly_correlated):
        # Here, we only model the extreme cases of the Gaussian process being perfectly correlated
        # (then perfectly_correlated = True) or perfectly uncorrelated (then perfectly_correlated = False)
        self.process_mean = process_mean
        self.process_variance = process_variance
        self.perfectly_correlated = perfectly_correlated
        self.realization_count = 8  # Match the numpy version

    # Compute the distribution of a random variable that is the sum of the integration of a gaussian process
    # over multiple intervals from t_a to t_b. Needs to take into account that the integrals are correlated.
    def get_sum_of_interval_integrals_mean_variance(self, t_a, t_b, mask):
        # t_a and t_b have shape (:, interval_count)
        total_delta_t = torch.sum(torch.where(
            mask, torch.abs(t_b - t_a), 0), axis=1)
        integrated_mean = self.process_mean * total_delta_t

        if self.perfectly_correlated:
            integrated_variance = total_delta_t**2 * self.process_variance
        else:
            integrated_variance = total_delta_t * self.process_variance

        return integrated_mean, integrated_variance

    def get_realization(self, t, variance_scale):
        std = math.sqrt(self.process_variance * variance_scale)
        if self.perfectly_correlated:
            # One sample, broadcast to (1, len(t))
            sample = torch.normal(self.process_mean, std, size=(1,))
            return sample * torch.ones((1, len(t)), dtype=torch.float32, device=t.device)
        else:
            # realization_count samples, each of length len(t)
            return torch.abs(torch.normal(self.process_mean, std, size=(self.realization_count, len(t)), device=t.device))


class VelocityDissimilarityMetric(GaussianDissimilarityMetric):
    def __init__(self, velocity_mean, velocity_variance, perfectly_correlated, timestamps):
        # Model absolute value of velocity (speed) as Gaussian process
        self.velocity_model = SimpleGaussianProcess(
            velocity_mean, velocity_variance, perfectly_correlated)
        self.timestamps = timestamps

    def get_realization(self, output_matrix, variance_scale):
        velocities = self.velocity_model.get_realization(
            self.timestamps[:-1], variance_scale).flatten()
        displacements = torch.cat([
            torch.zeros(1, dtype=velocities.dtype, device=velocities.device),
            torch.cumsum(velocities * torch.diff(self.timestamps).flatten(), dim=0)
        ])
        output_matrix[:] = torch.abs(displacements[None, :] - displacements[:, None])

        # Numerical trick that makes shortest path algorithm "skip" unnecessary intermediary hops:
        # Add a tiny additional cost to each hop. This way, e.g. path A->C is cheaper than path A->B->C
        # Since this reduces the overall path length (and hence also the length of the longest shortest path),
        # this makes path generation much faster later on.
        # (Since CC runs shortest path algorithm on neighborhood graph, not all intermediary nodes will be skipped.
        # This is only achieved later through path contraction.)
        path_hop_cost = torch.ones_like(output_matrix) * 1e-5
        path_hop_cost.fill_diagonal_(0)
        output_matrix[:] += path_hop_cost

    def mean_variance_along_path(self, paths, mask):
        t_a = self.timestamps[paths[:, :-1]]
        t_b = self.timestamps[paths[:, 1:]]

        return self.velocity_model.get_sum_of_interval_integrals_mean_variance(t_a, t_b, mask)

    def is_contractable(self):
        return True

    def get_datapoint_count(self):
        return self.timestamps.shape[0]
