import multiprocessing as mp
import numpy as np
from quantKit.stats.stat_helpers import mutual_info
from quantKit.stats.mcpt.BatchCyclicPermutation import bcp

def generate_mi_report(
    features: np.recarray, 
    target: np.recarray, 
    nbins_feature: int = 10, 
    nbins_target: int = 10, 
    n_permutations: int = 100
) -> None:
    """
    Generates a report with mutual information, solo p-values, and unbiased p-values.

    Parameters:
    - features (np.recarray): Record array of features information for mutual information calculations.
    - target (np.recarray): The target data, typically the 'close' price of a trade-bar.
    - nbins_feature (int): Number of bins for discretizing the features. Default is 10.
    - nbins_target (int): Number of bins for discretizing the target. Default is 10.
    - n_permutations (int): Number of permutations for the target data. Default is 100.
    """
    feature_fields = [field for field in features.dtype.names if field != 'Date']
    target_fields = [field for field in target.dtype.names if field != 'Date']

    original_mi_scores = []

    # Calculate original mutual information scores
    for feature_field in feature_fields:
        for target_field in target_fields: 
            mi_score = mutual_info(
                features[feature_field], 
                target[target_field], 
                nbins_feature=nbins_feature, 
                nbins_target=nbins_target
            )
            original_mi_scores.append((feature_field, target_field, mi_score))

    original_mi_scores = np.array([score[2] for score in original_mi_scores])

    target_permutations = []

    # Generate permutations for each target field
    for target_field in target_fields:
        target_permutations.append(bcp(target[target_field], n_permutations))

    # Convert to a NumPy array for easier handling
    target_permutations = np.array(target_permutations)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        permuted_mi_scores = pool.starmap(
            mutual_info,
            [
                (features[feature_field], permuted_target, nbins_feature, nbins_target)
                for permuted_targets in target_permutations
                for permuted_target in permuted_targets
                for feature_field in feature_fields
            ]
        )

    permuted_mi_scores = np.array(permuted_mi_scores).reshape(len(feature_fields), len(target_fields), n_permutations)

    # Calculate p-values
    solo_p_values = np.mean(permuted_mi_scores >= original_mi_scores[:, None, None], axis=2)
    unbiased_p_values = (np.sum(permuted_mi_scores >= original_mi_scores[:, None, None], axis=2) + 1) / (n_permutations + 1)

    # Print results
    print("## Mutual Information Report")
    print()
    print("High MI scores indicate a strong relationship between the indicator and the target variable, suggesting potential predictive power. Low p-values further validate the indicator's statistical significance.")
    print()
    print("**MI Score**: Measures the mutual dependence between the feature and the target.")
    print("**Solo p-value**: Initial significance estimate, proportion of permuted MI scores equal to or higher than the original MI scores.")
    print("**Unbiased p-value**: Adjusted solo p-value considering the number of permutations plus one, reducing bias.")
    print()
    print("| Indicator           | Target              | MI Score            | Solo p-value        | Unbiased p-value    |")
    print("|---------------------|---------------------|---------------------|---------------------|---------------------|")

    for (feature_field, target_field), mi_score, solo_p, unbiased_p in zip(
            [(f, t) for f in feature_fields for t in target_fields],
            original_mi_scores,
            solo_p_values.flatten(),
            unbiased_p_values.flatten()):
        print(f"| {feature_field:<18} | {target_field:<18} | {mi_score:<19.4f} | {solo_p:<19.4f} | {unbiased_p:<19.4f} |")

