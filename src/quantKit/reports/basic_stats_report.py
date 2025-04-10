import numpy as np
from quantKit.stats.stat_helpers import simple_stats, iqr, range_iqr_ratio, relative_entropy

# Updated basic statistics report function
def generate_basic_stats_report(data) -> None:
    """
    Generate a basic statistical report for each column in the provided recarray.

    Parameters
    ----------
    data : np.recarray
        Record array containing numerical data for which to generate the report.

    Returns
    -------
    None
        The function prints the basic statistics report directly to the console.
    """
    if not isinstance(data, np.recarray):
        raise ValueError("Input data must be a numpy recarray.")

    results = []

    for column in data.dtype.names:
        if column == 'Date':
            continue
        values = data[column]
        values = values[~np.isnan(values)]  # Remove NaN values
        
        # Calculate basic statistics
        ncases, mean, min_value, max_value = simple_stats(values)
        interquartile_range = iqr(values)
        rnq_iqr = range_iqr_ratio(values)
        entropy_value = relative_entropy(values)
        
        results.append({
            'Indicator': column,
            'ncases': ncases,
            'mean': mean,
            'min': min_value,
            'max': max_value,
            'iqr': interquartile_range,
            'rnq/IQR': rnq_iqr,
            'relative_entropy': entropy_value
        })

    print("## Simple Statistics and Relative Entropy Report")
    print()
    print("The Simple Statistics Table summarizes key metrics for each trading indicator, including the number of cases, mean, minimum, maximum, interquartile range (IQR), range/IQR ratio, and relative entropy. In the table, a lower range/IQR ratio suggests a tighter, more predictable dataset, while an optimal relative entropy indicates a balance of diversity and uniqueness without excessive noise.")
    print()
    print("**Ncases**: Number of cases (bars) in feature (indicator).")
    print("**Mean**: Average value of the feature across all cases.")
    print("**Min/Max**: The minimum and maximum value of the feature across all cases.")
    print("**IQR**: Interquartile Range, measures range minus the top and bottom 25% of the raw range.")
    print("**Range/IQR**: A unitless measure of data dispersion relative to its middle 50%.")
    print("**Relative Entropy**: Measures the difference between two probability distributions; a value of zero indicates identical distributions.")
    print()
    print("| Indicator           | Ncases | Mean           | Min            | Max            | IQR            | rnq/IQR        | Relative Entropy    |")
    print("|---------------------|--------|----------------|----------------|----------------|----------------|----------------|---------------------|")
    
    for result in results:
        print(f"| {result['Indicator']:<19} | {result['ncases']:<6} | {result['mean']:<14.4f} | {result['min']:<14.4f} | {result['max']:<14.4f} | {result['iqr']:<14.4f} | {result['rnq/IQR']:<14.4f} | {result['relative_entropy']:<19.4f} |")

