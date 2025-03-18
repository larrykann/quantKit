import os
import matplotlib
import matplotlib.offsetbox as offsetbox
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional

# Set Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")

def save_plot_to_file(
    features: np.recarray,
    images_directory: str,
    logo_path: Optional[str] = None
) -> None:
    """
    Saves plots of all indicators in the given recarray over time to a specified directory.
    
    Parameters:
    - features: recarray containing the data to be plotted.
    - images_directory (str): Directory where the images will be saved.
    - logo_path (Optional[str]): Path to the logo image to be added to the plot (default is None).
    """
    if not images_directory:
        raise ValueError("'images_directory' must be provided.")

    # Set backend to Agg to avoid GUI output
    matplotlib.use('Agg')

    columns_to_plot = [column for column in features.dtype.names if column != 'Date']

    for column in columns_to_plot:
        plt.figure(figsize=(15, 6), dpi=300)  # High DPI for sharp images
        plt.plot(features['Date'], features[column], label=column, color='black', linewidth=0.5)
        plt.title(f"{column} over Time", fontsize=16, weight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel(column, fontsize=14)

        if logo_path:
            logo = plt.imread(logo_path)
            imagebox = offsetbox.AnchoredOffsetbox(loc="lower right", child=offsetbox.OffsetImage(logo, zoom=0.1), pad=0.5, frameon=False)
            plt.gca().add_artist(imagebox)

        plt.legend()
        image_path = os.path.join(images_directory, f"{column}.png")
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

