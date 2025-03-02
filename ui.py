import matplotlib.pyplot as plt
import pandas as pd
import sys
import re
from colorama import init, Fore, Style

# Initialize colorama (needed for Windows)
init(autoreset=True)
# Define custom color mapping
COLOR_PALETTE = {
    "col": Fore.CYAN,
    "missing_urs": Fore.RED,
    "option": Fore.RED,
    "first_input": Fore.YELLOW,
    "op": Fore.YELLOW,
    "second_input": Fore.YELLOW,
    "unit": Fore.CYAN
}

def styled_print(text, color="white", style="normal", end="\n", **kwargs):
    """
    Prints text with inline styling, automatically detecting and formatting variables.

    Parameters:
        text (str): The text to print, containing `{}` placeholders or just static text.
        color (str): Default text color (only applies if no custom variable colors are used).
        style (str): Text style. Options: 'bold', 'dim', 'normal'.
        end (str): End character (default is newline).
        **kwargs: Optional manually passed variables (not needed if using locals()).

    Example Usage:
        styled_print("Baseline Demographic Stats:", color="blue", style="bold")
        styled_print("For column '{col}', the following UR values have missing data: {missing_urs}")
    """
    colors = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE
    }
    
    styles = {
        "bold": Style.BRIGHT,
        "dim": Style.DIM,
        "normal": Style.NORMAL
    }

    color_code = colors.get(color.lower(), Fore.WHITE)
    style_code = styles.get(style.lower(), Style.NORMAL)

    # Auto-detect variables inside text (matching `{variable}`)
    detected_vars = re.findall(r"\{(.*?)\}", text)

    if detected_vars:
        # Get local variables dynamically
        caller_locals = sys._getframe(1).f_locals  # Gets the calling function's variables

        # Replace detected variables with their values and apply colors
        formatted_kwargs = {}
        for var in detected_vars:
            if var in caller_locals:  # Check if the variable exists in the caller's local scope
                value = caller_locals[var]
                if var in COLOR_PALETTE:  # Apply color if in the palette
                    formatted_kwargs[var] = f"{COLOR_PALETTE[var]}{value}{Style.RESET_ALL}"
                else:  # Render normally if not in the palette
                    formatted_kwargs[var] = value

        # Format text using dynamically found variables
        formatted_text = text.format(**formatted_kwargs)

        sys.stdout.write(f"{style_code}{formatted_text}{Style.RESET_ALL}{end}")

    else:
        # No variables, just format and print the static text
        sys.stdout.write(f"{style_code}{color_code}{text}{Style.RESET_ALL}{end}")

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