import multiprocessing as mp
import numpy as np
from quantKit.stats.stat_helpers import compute_serial_correlated_break as mb

def generate_mcmbt_report(
    data: np.recarray, 
    min_recent: int = 100, 
    max_recent: int = 500, 
    lag: int = 1, 
    n_permutations: int = 100
) -> None:
    """
    Executes serial correlation break tests for each variable in the dataset using specified parameters.
    Permutes data to compute p-values for detected breaks via Monte Carlo simulations.

    Parameters:
        data (np.recarray): A record array where each column/field represents a variable.
        min_recent (int): Minimum size of the most recent subset to consider for the break test.
        max_recent (int): Maximum size of the most recent subset to consider for the break test.
        lag (int): Number of lags (serial correlation) to consider in the test.
        n_permutations (int): Number of permutations to use for the Monte Carlo simulation to compute p-values.

    Prints:
        The results of the break tests, including:
            - Indicator: Name of the variable.
            - nrecent: The break point detected in the data.
            - z(U): The maximum critical value of the test statistic.
            - Solo p-value: P-value computed for this variable alone.
            - Unbiased p-value: P-value adjusted for multiple testing (if applicable).
    """
    var_names = [name for name in data.dtype.names if name != 'Date']
    n_cases = len(data)
    n_vars = len(var_names)

    if n_vars == 0:
        print("Error: No indicators found in the recarray. Maybe it's time to reconsider your data sources?")
        return

    # Initialize results recarray
    results_dtype = [
        ('Indicator', 'U50'),
        ('nrecent', 'f8'),
        ('z(U)', 'f8'),
        ('Solo p-value', 'f8'),
        ('Unbiased p-value', 'f8')
    ]
    results = np.zeros(n_vars, dtype=results_dtype)

    results['Indicator'] = var_names

    # Compute original max critical values and breakpoints
    original_results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        original_results = pool.starmap(
            mb,
            [(data[var].astype(np.float64), n_cases, min_recent, max_recent, lag) for var in var_names]
        )

    # Populate results with original computations
    original_max_crits = np.array([res[0] for res in original_results])
    n_recent_values = np.array([res[1] for res in original_results])

    results['nrecent'] = n_recent_values
    results['z(U)'] = original_max_crits

    # Initialize solo p-values
    solo_p_values = np.zeros(n_vars)

    # Perform permutations and compute permuted max critical values
    permuted_max_crits = np.zeros((n_permutations, n_vars))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i in range(n_permutations):
            permuted_data = {
                var: np.random.permutation(data[var]) for var in var_names
            }
            permuted_results = pool.starmap(
                mb,
                [(permuted_data[var].astype(np.float64), n_cases, min_recent, max_recent, lag) for var in var_names]
            )
            permuted_crits = np.array([res[0] for res in permuted_results])
            permuted_max_crits[i] = permuted_crits

    # Calculate solo p-values
    for idx in range(n_vars):
        solo_p_values[idx] = np.mean(permuted_max_crits[:, idx] >= original_max_crits[idx])

    results['Solo p-value'] = solo_p_values

    # Calculate unbiased p-value (global test) if more than one variable
    if n_vars > 1:
        global_original_max = np.max(original_max_crits)
        global_permuted_max = np.max(permuted_max_crits, axis=1)
        unbiased_p_value = np.mean(global_permuted_max >= global_original_max)
        results['Unbiased p-value'] = unbiased_p_value
    else:
        results['Unbiased p-value'] = np.nan

    # Print the results
    print("## Serial Correlated Mean Break Test Report")
    print()
    print("The Serial Correlated Mean Break Test Report identifies potential breaks in the mean of each trading indicator, taking into account serial correlation. This test helps detect significant shifts in the mean over time, indicating nonstationary behavior in the data.")
    print()
    print("**nrecent**: The number of recent observations considered in the test.")
    print("**z(U)**: The greatest break encountered in the mean across the user-specified range.")
    print("**Solo p-value**: Measures the significance of the greatest break while accounting for the entire range of boundaries searched. If this value is not small, it suggests that the indicator does not have a significant mean break.")
    if n_vars > 1:
        print("**Unbiased p-value**: Adjusted p-value considering multiple indicators.")
    print()
    header = "| Indicator           | n_recent | z(U)     | Solo p-value |"
    separator = "|---------------------|----------|----------|--------------|"
    if n_vars > 1:
        header += " Unbiased p-value |"
        separator += "-----------------|"
    print(header)
    print(separator)
    for row in results:
        line = f"| {row['Indicator']:<19} | {row['nrecent']:<8} | {row['z(U)']:<8.4f} | {row['Solo p-value']:<12.4f} |"
        if n_vars > 1:
            line += f" {row['Unbiased p-value']:<15.4f} |"
        print(line)

