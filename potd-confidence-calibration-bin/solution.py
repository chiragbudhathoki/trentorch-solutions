def confidence_calibration_bin(
    probs: list[float], labels: list[int], k: int
) -> list[tuple[int, float, float]]:
    """
    probs: predicted probabilities, each in [0, 1].
    labels: binary true labels (0 or 1), same length as probs.
    k: number of equal-width bins.

    bin(p) = min(k - 1, floor(k * p))

    For each of the k bins (in order 0..k-1), return
    (count, mean_prob, positive_rate). For an empty bin, mean_prob and
    positive_rate are both 0.0.
    """
    # TODO: assign each point to a bin with the clamped formula, then
    # aggregate per bin
    buckets = [[]for _ in range(k)]
    for p,y in zip(probs,labels):
        bin_idx = min(k -1,int(k * p))
        buckets[bin_idx].append((p,y))
    result = []
    for bucket in buckets:
        count = len(bucket)
        if count == 0:
          result.append((0,0.0,0.0))
        else:
            mean_prob =  sum(p for p,_ in bucket)/count
            positive_rate = sum(y for _,y in bucket)/count
            result.append((count,mean_prob,positive_rate))
        
    return (result)
