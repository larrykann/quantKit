import os
import csv
from datetime import datetime

import numpy as np
from quantKit.stats.stat_helpers import relative_entropy
from rich.console import Console
from rich.table import Table

# Initialize a shared Rich concole
console = Console()

def generate_basic_stats_report(
    data: np.recarray,
    save_csv: bool = False,
    csv_dir: str = None
) -> None:
    """
    Generate a basic statistical report for each column in the provided recarray,
    display it in the terminal using Rich, and optionally save to CSV.

    Parameters
    ----------
    data : np.recarray
        Record array containing numerical data for which to generate the report.
    save_csv : bool, default False
        If True, save the results to a CSV file.
    csv_dir : str, optional
        Directory to save the CSV file in. Defaults to 'Results'.
    """
    # Validate input
    if not isinstance(data, np.recarray):
        raise ValueError("Input data must be a numpy recarray.")

    results = []
    for col in data.dtype.names:
        if col == 'Date':
            continue
        values = data[col]
        # Drop NaNs
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue

        # Basic stats inlined
        ncases = values.size
        mean = float(np.mean(values))
        min_value = float(np.min(values))
        max_value = float(np.max(values))

        # IQR calculation
        q1, q3 = np.percentile(values, [25, 75], method='midpoint')
        interquartile_range = float(q3 - q1)

        # Range/IQR calculation
        rnq_iqr = float((max_value - min_value) / (interquartile_range + 1e-60))

        # Relative entropy
        entropy_value = float(relative_entropy(values))

        results.append({
            'Indicator': col,
            'N': ncases,
            'Mean': mean,
            'Min': min_value,
            'Max': max_value,
            'IQR': interquartile_range,
            'Range/IQR': rnq_iqr,
            'Rel. Entropy': entropy_value
        })

    # Build Rich table
    table = Table(
        title="Basic Statistics Report",
        show_header=True,
        header_style="bold magenta"
    )
    headers = ['Indicator', 'N', 'Mean', 'Min', 'Max', 'IQR', 'Range/IQR', 'Rel. Entropy']
    for header in headers:
        justify = 'right' if header != 'Indicator' else 'left'
        table.add_column(header, justify=justify)

    for row in results:
        table.add_row(
            row['Indicator'],
            f"{row['N']:,}",
            f"{row['Mean']:.4f}",
            f"{row['Min']:.4f}",
            f"{row['Max']:.4f}",
            f"{row['IQR']:.4f}",
            f"{row['Range/IQR']:.4f}",
            f"{row['Rel. Entropy']:.4f}"
        )

    # Render the table
    console.print(table)

    # Optional CSV saving
    if save_csv and results:
        out_dir = csv_dir or 'Results'
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{timestamp}_basic_stats.csv"
        path = os.path.join(out_dir, filename)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        console.print(f"[bold green]✓ Saved CSV to:[/bold green] {path}")


if __name__ == '__main__':
    # Example usage (replace with your recarray load):
    # data = np.load('your_data.npy', allow_pickle=True)
    # generate_basic_stats_report(data, save_csv=True)
    pass


