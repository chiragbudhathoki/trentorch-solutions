import numpy as np
from scipy.special import log_softmax


def class_weighted_loss(
    logits: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    """
    logits: n x C logits.
    labels: length-n 0-based true class indices.
    weights: length-C per-class weights.

    p_ic = softmax(logits_i)_c
    L = -(1/n) * sum_i weights[labels[i]] * log(p_i, labels[i])

    Divide by n, NOT by sum(weights[labels]). Return L.
    """
    # TODO: stable log-softmax per row, weight by true-class weight,
    # sum, divide by n.
    n = len(labels)
    log_probs = log_softmax(logits,axis = 1)
    true_class_log_probs = log_probs[np.arange(n),labels]
    sample_weights  = weights[labels]
    L = -np.sum(sample_weights * true_class_log_probs)/n
    return float(L)
