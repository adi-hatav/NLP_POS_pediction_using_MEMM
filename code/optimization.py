import numpy as np
import pickle
from preprocessing import represent_input_with_features, FeatureStatistics, Feature2id
from scipy.optimize import fmin_l_bfgs_b


def calc_objective_per_iter(w_i, *args):
    """
    Calculate max entropy likelihood for an iterative optimization method
    @param w_i: weights vector in iteration i
    @param args: arguments passed to this function, such as lambda hyper-parameter for regularization

    @return: the Max Entropy likelihood (objective) and the objective gradient
    """
    all_histories_tuples, represent_input_with_features_function, feature2id, lam = args
    tags = feature2id.feature_statistics.tags

    linear_term = (feature2id.small_matrix @ w_i).sum()

    helper = np.exp(feature2id.big_matrix @ w_i).reshape(-1, len(tags))
    normalization_term = np.log(helper.sum(axis=1)).sum()
    regularization = 0.5 * lam * (np.linalg.norm(w_i) ** 2)

    empirical_counts = feature2id.small_matrix.sum(axis=0)

    divider = (helper.sum(axis=1).reshape(-1, 1) @ np.ones(len(tags)).reshape(1, -1)).reshape(-1)
    mat_B = helper.reshape(-1) / divider
    expected_counts = feature2id.big_matrix.T.multiply(mat_B).T.sum(axis=0)

    regularization_grad = w_i * lam

    likelihood = linear_term - normalization_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad

    return (-1) * likelihood, (-1) * grad


def get_optimal_vector(statistics: FeatureStatistics, feature2id: Feature2id, lam: float, weights_path: str) -> None:
    """
    The function computes and saves to a file the optimal weights
    @param statistics: The Feature Statistics object containing the histories and their tags
    @param feature2id: The Feature2ID object
    @param lam: the regularizer lambda to use for the L2 loss in the optimization
    @param weights_path: the path in which to save the optimal weights

    """
    args = (statistics.histories, represent_input_with_features, feature2id, lam)
    w_0 = np.random.normal(0, 1, feature2id.n_total_features)

    optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter,
                                   x0=w_0,
                                   args=args,
                                   maxiter=750,
                                   iprint=10,
                                   epsilon=1e-7,
                                   bounds=None)
    with open(weights_path, 'wb+') as f:
        pickle.dump((optimal_params, feature2id), f)
