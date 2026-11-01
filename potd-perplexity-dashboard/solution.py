import math
import statistics

def perplexity_dashboard(losses: list[float]) -> tuple[float, float]:
    """
    losses: a list of N per-token negative log-likelihood values.

    H = mean(losses)
    PPL = e^H

    Return (mean_loss, perplexity). Exponentiate the MEAN, never the sum.
    """
    # TODO: average first, then exponentiate once.
    H = statistics.mean(losses)
    PPL = math.exp(H)
    return(H,PPL)
