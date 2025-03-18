import numpy as np

def bcp(target: np.ndarray, n_permutations: int) -> np.ndarray:
    """
    Generates cyclic permutations of the target array.

    Parameters:
    - target (np.ndarray): The target data array to be permuted.
    - n_permutations (int): Number of permutations to generate.

    Returns:
    - np.ndarray: An array of permuted target arrays.
    """
    n = len(target)
    target_permutations = np.empty((n_permutations, n), dtype=target.dtype)
    for i in range(n_permutations):
        shift = np.random.randint(0, n)  # Generate a single shift value
        target_permutations[i] = np.roll(target, shift)
    
    return target_permutations
