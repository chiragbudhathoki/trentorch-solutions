import numpy as np
from scipy.special import softmax


def temperature_calibrator(z: np.ndarray, tau: float) -> np.ndarray:
    """
    z: logits.
    tau: temperature, tau > 0.

    p = softmax(z / tau), using an overflow-safe softmax (subtract the
    max before exponentiating). Divide by tau, never multiply.

    Return p.
    """
    # TODO: scale by dividing, then stable softmax.
    if tau <= 0:
        raise ValueError(
            f"Temperature 'tau' must be strictly positive, got {tau}."
        )
    p = softmax(z / tau,axis = -1)
    return p
