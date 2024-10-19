import plotext as plx

def plot_to_terminal(recarray, hline_label='Zero', hline_value=0):
    """
    Prints plots of all indicators in the given recarray to the terminal.
    
    Parameters:
    - recarray: recarray containing the data to be plotted.
    - hline_label: Label for the horizontal line (default is 'Zero').
    - hline_value: Value for the horizontal line (default is 0).
    """
    columns_to_plot = [column for column in recarray.dtype.names if column != 'Date']

    for column in columns_to_plot:
        plx.clear_figure()
        plx.plot(recarray['Date'], recarray[column], label=column)
        plx.title(f"{column} over Time")
        plx.xlabel('Date')
        plx.ylabel(column)
        plx.show()

