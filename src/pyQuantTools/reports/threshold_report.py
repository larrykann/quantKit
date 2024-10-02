import pandas as pd
from pyQuantTools.stats.threshold.threshold_table import generate_threshold_table
from pyQuantTools.stats.threshold.threshold_opt import opt_thresh
from pyQuantTools.stats.mcpt.threshold_mcpt import opt_MCPT

def generate_threshold_report(
    features: pd.DataFrame,
    target: pd.DataFrame,
    bins: int = 13,
    min_cases_percent: int = 5,
    n_permutations: int = 100
) -> None:
    """
    Generates a threshold report with profit factors above and below various thresholds,
    and calculates p-values using Monte Carlo permutation testing (MCPT).

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing indicator (signal) values. Columns represent different indicators.
    target : pd.DataFrame
        DataFrame containing target return values. Columns represent different return metrics.
    bins : int, optional
        Number of bins for threshold calculations. Must be either 13 or 27. Default is 13.
    min_cases_percent : int, optional
        Minimum percentage of cases required for threshold calculation to ensure statistical significance. 
        Default is 5%.
    n_permutations : int, optional
        Number of permutations for Monte Carlo Permutation Testing (MCPT). 
        Setting to 0 disables permutation testing and p-values will not be calculated. 
        Default is 100.

    Returns
    -------
    None
        The function prints the threshold report directly to the console.

    Raises
    ------
    ValueError
        If `features` or `target` is not a pandas DataFrame.
        If `bins` is not either 13 or 27.
        If `min_cases_percent` is not between 0 and 100.
        If `n_permutations` is negative.

    Example
    -------
    ```python
    import pandas as pd
    import numpy as np
    from your_library_name.reports.threshold_report import generate_threshold_report

    # Example data
    np.random.seed(42)  # For reproducibility
    features = pd.DataFrame({
        'Indicator1': np.random.rand(100),
        'Indicator2': np.random.rand(100),
        'Date': pd.date_range(start='2020-01-01', periods=100)
    })

    target = pd.DataFrame({
        'Return': np.random.randn(100),
        'Date': pd.date_range(start='2020-01-01', periods=100)
    })

    # Generate the threshold report
    generate_threshold_report(
        features=features,
        target=target,
        bins=13,
        min_cases_percent=5,
        n_permutations=100
    )
    ```
    """
    
    # Validate input types
    if not isinstance(features, pd.DataFrame):
        raise ValueError("Features must be a pandas DataFrame.")
    if not isinstance(target, pd.DataFrame):
        raise ValueError("Target must be a pandas DataFrame.")
    
    # Validate 'bins' parameter
    if bins not in [13, 27]:
        raise ValueError("Bins must be either 13 or 27.")
    
    # Validate 'min_cases_percent' parameter
    if not (0 <= min_cases_percent <= 100):
        raise ValueError("min_cases_percent must be between 0 and 100.")
    
    # Validate 'n_permutations' parameter
    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative.")
    
    # Remove 'Date' column if present
    feature_fields = features.drop(columns=['Date'], errors='ignore')
    target_fields = target.drop(columns=['Date'], errors='ignore')

    header_printed = False

    for feature_field in feature_fields.columns:
        feature = feature_fields[feature_field].values

        for target_field in target_fields.columns:
            target_array = target_fields[target_field].values

            # Generate ROC table using the updated function
            roc_table = generate_threshold_table(signal_vals=feature, returns=target_array, bin_count=bins)

            if not header_printed:
                print("## Optimal Thresholds w/ Profit Factor Report")
                print()
                print("The Optimal Thresholds w/ Profit Factor Report evaluates various threshold levels for trading indicators to identify the most profitable long and short positions. The report includes the fraction of data points greater than or equal to the threshold, the corresponding profit factor for long and short positions, and the fraction of data points less than the threshold with their respective profit factors. The optimal thresholds at the bottom indicate the threshold levels with the highest profit factors for long and short positions, while the p-values provide statistical significance for these thresholds.")
                header_printed = True

            print()
            print(f"### {feature_field} vs {target_field}")
            print()
            print("| Threshold | Frac Gtr/Eq | Long PF    | Short PF   | Frac Less | Short PF   | Long PF    |")
            print("|-----------|-------------|------------|------------|-----------|------------|------------|")

            for row in roc_table:
                print(f"| {row[0]:8.3f} | {row[1]:10.3f} | {row[2]:12.4f} | {row[3]:12.4f} | {row[4]:13.3f} | {row[5]:12.4f} | {row[6]:12.4f} |")

            # Calculate minimum number of cases as a percentage of total cases
            n = len(feature)
            min_kept = max(1, int(min_cases_percent * n / 100))

            # Call the updated opt_thresh function
            pf_all, high_thresh, pf_high, low_thresh, pf_low = opt_thresh(
                min_cases_percent=min_cases_percent,
                predictor=feature,
                target=target_array,
                use_log=False
            )

            print()
            if pf_all > 1e20:
                print("**Grand profit factor**: infinite")
            else:
                print(f"**Grand profit factor**: {pf_all:.3f}")

            if pf_high > 1e20:
                print(f"**Optimal long threshold**: {high_thresh:.4f}, profit factor = infinite")
            else:
                print(f"**Optimal long threshold**: {high_thresh:.4f}, profit factor = {pf_high:.3f}")

            if pf_low > 1e20:
                print(f"**Optimal short threshold**: {low_thresh:.4f}, profit factor = infinite")
            else:
                print(f"**Optimal short threshold**: {low_thresh:.4f}, profit factor = {pf_low:.3f}")

            if n_permutations > 0:
                # Perform MCPT using the updated opt_MCPT function
                (
                    pf_all_mcpt,
                    high_thresh_mcpt,
                    pf_high_mcpt,
                    low_thresh_mcpt,
                    pf_low_mcpt,
                    pval_long,
                    pval_short,
                    pval_best
                ) = opt_MCPT(
                    signal_vals=feature,
                    returns=target_array,
                    min_kept=min_kept,
                    flip_sign=False,
                    nreps=n_permutations
                )

                print()
                print(f"**P-values**: Long={pval_long:.3f}, Short={pval_short:.3f}, Best={pval_best:.3f}")

