def bonferroni(
    pvalues: list[float], alpha: float = 0.05, use_holm: bool = True
) -> bool:
    m = len(pvalues)
    for i, pvalue in enumerate(sorted(pvalues)):
        if pvalue < alpha / (m - i * use_holm):
            continue
        return False
    return True
