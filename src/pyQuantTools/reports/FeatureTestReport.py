import os
from contextlib import redirect_stdout
from datetime import datetime
from pyQuantTools.visualizations.SavePlotToFile import save_plot_to_file
from pyQuantTools.reports.mutual_info_report import generate_mi_report
from pyQuantTools.reports.threshold_report import generate_threshold_report
from pyQuantTools.reports.basic_stats_report import generate_basic_stats_report
from pyQuantTools.reports.mcmbt_report import generate_mcmbt_report

def run_indicator_tests(features, target, report_name, file_path=None, file_extension='md', **kwargs):
    """
    Runs a suite of indicator tests on the given features dataframe and target, and generates a report.

    This function performs the following tests:
    1. Basic Statistics Report
    2. Mutual Information Report
    3. Serial Correlated Mean Break Test Report
    4. Optimal Thresholds with Profit Factor Report

    The report is saved as a Markdown (.md) file with a dynamically generated filename that includes the report name and a timestamp.

    Parameters:
    ----------
    features : np.recarray
        Record array containing the indicator values.

    target : np.recarray
        Record array containing the target values (e.g., returns).

    report_name : str
        Name of the report, used in the filename.

    file_path : str, optional
        Directory path for saving the output report. If None, defaults to 'IndicatorReports/' in the current working directory.

    file_extension : str, optional
        File extension for the output report. Either 'txt' or 'md'. Defaults to 'md'.

    save_plots_to_file : bool, optional
        Whether to save plots as images to the file path or print plots to the terminal (default is True).

    kwargs : dict, optional
        Additional keyword arguments for customizing the parameters of specific tests. Supported keys are:
        - 'statistics_params': dict, parameters for the basic statistics report (default: empty dict).
        - 'mi_params': dict, parameters for the mutual information report (default: {'n_permutations': 100}).
        - 'mcmbt_params': dict, parameters for the serial correlated mean break test report (default: {'min_recent': 100, 'max_recent': 5000, 'lag': 1, 'n_permutations': 100}).
        - 'threshold_params': dict, parameters for the optimal thresholds with profit factor report (default: {'bins': 13, 'min_cases': 5, 'use_mcpt': True, 'n_permutations': 100}).

    Returns:
    -------
    None

    Saves the report as a Markdown (.md) or Text (.txt) file in the specified or default directory.

    """

    # Ensure the file_extension is either 'txt' or 'md'
    if file_extension not in ['txt', 'md']:
        raise ValueError("file_extension must be either 'txt' or 'md'")
        
    # Default parameters for each test
    statistics_params = kwargs.get('statistics_params', {})
    mi_params = kwargs.get('mi_params', {'n_permutations': 100})
    mcmbt_params = kwargs.get('mcmbt_params', {'min_recent': 100, 'max_recent': 5000, 'lag': 1, 'n_permutations': 100})
    threshold_params = kwargs.get('threshold_params', {'bins': 13, 'min_cases': 5, 'use_mcpt': True, 'n_permutations': 100})

    # Ensure the directory exists
    if file_path is None:
        # Default to the IndicatorReports directory in the project root
        base_directory = os.path.join(os.getcwd(), 'IndicatorReports')
    else:
        base_directory = file_path

    # Create directory for indicator report
    report_directory = os.path.join(base_directory, report_name)
    os.makedirs(report_directory, exist_ok=True)

    # Create the images directory
    images_directory = os.path.join(report_directory, 'images')
    os.makedirs(images_directory, exist_ok=True)

    # Create a dynamic filename with a datetime stamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{report_name}_{timestamp}.{file_extension}"
    file_path = os.path.join(report_directory, filename)

    # Open the output file in write mode
    with open(file_path, 'w') as f:
        # Redirect stdout to the file
        with redirect_stdout(f):
            print(f"# Indicator Soundness Report {report_name}")
            print("The generated report provides a comprehensive analysis of trading indicators, offering insights into their statistical properties, predictive power, mean stability over time, and optimal thresholds for profitability. It combines detailed statistical summaries, mutual information scores, mean break tests, and profit factor evaluations.")
            print()
            save_plot_to_file(features, images_directory)
            save_plot_to_file(target, images_directory)
            print()
           
            for column in features.dtype.names:
                image_path = os.path.join(images_directory, f"{column}.png")
                print(f"![{column}]({image_path})")
                print()
            for column in target.dtype.names:
                image_path = os.path.join(images_directory, f"{column}.png")
                print(f"![{column}]({image_path})")
                print()

            generate_basic_stats_report(features, **statistics_params)
            generate_mi_report(features, target, **mi_params)
            print()
            generate_mcmbt_report(features, **mcmbt_params)
            print()
            generate_threshold_report(features, target, **threshold_params)
    
    print(f"Report has been saved to {file_path}")

