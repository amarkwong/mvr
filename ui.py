import matplotlib.pyplot as plt
import pandas as pd

def plot_histogram(stat_dict, xlabel, ylabel, title):
    """
    Plot a histogram from a dictionary of statistics.
    """
    keys = list(stat_dict.keys())
    values = list(stat_dict.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(keys, values, color='skyblue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def print_table_markdown(table_df):
    """
    Print a pandas DataFrame as a Markdown table.
    Requires pandas >= 1.0 for DataFrame.to_markdown().
    """
    try:
        md_table = table_df.to_markdown(index=False)
    except AttributeError:
        md_table = table_df.to_string(index=False)
    print(md_table)