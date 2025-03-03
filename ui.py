import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import re
import statsmodels.api as sm
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

def ols_to_markdown(results):
    """
    Converts statsmodels OLS regression results to a Markdown table format.

    Parameters:
        results (RegressionResults): Fitted OLS model from statsmodels

    Returns:
        str: Markdown-formatted table
    """

    # Extract summary statistics
    header = f"## OLS Regression Results\n\n"
    stats_table = f"""
| Metric               | Value        |
|----------------------|-------------|
| **Dependent Variable**  | {results.model.endog_names} |
| **R-squared**          | {results.rsquared:.3f} |
| **Adj. R-squared**     | {results.rsquared_adj:.3f} |
| **Method**             | OLS |
| **F-statistic**        | {results.fvalue:.3f} |
| **Prob (F-statistic)** | {results.f_pvalue:.3e} |
| **No. Observations**   | {results.nobs:.0f} |
| **AIC**                | {results.aic:.1f} |
| **BIC**                | {results.bic:.1f} |
"""

    # Extract coefficient table
    coef_df = pd.DataFrame({
        "Variable": results.params.index,
        "Coef": results.params.values,
        "Std Err": results.bse.values,
        "t": results.tvalues.values,
        "P>|t|": results.pvalues.values,  # Escape `>` later
        "[0.025": results.conf_int()[0].values,
        "0.975]": results.conf_int()[1].values
    })

    # Escape special characters in headers
    headers = ["Variable", "Coef", "Std Err", "t", "P&gt;\|t\|", "[0.025", "0.975]"]

    coef_table = "| " + " | ".join(headers) + " |\n"
    coef_table += "|-" + "-|-".join(["------"] * len(headers)) + "-|\n"

    for _, row in coef_df.iterrows():
        coef_table += f"| **{row['Variable']}** | {row['Coef']:.4f} | {row['Std Err']:.4f} | {row['t']:.3f} | {row['P>|t|']:.3f} | {row['[0.025']:.3f} | {row['0.975]']:.3f} |\n"

    # Model diagnostics (handle missing attributes safely)
    diagnostics_table = "\n### **Model Diagnostics**\n"

    if hasattr(results, "dwstat"):
        diagnostics_table += f"- **Durbin-Watson**: {results.durbin_watson:.3f}\n"
    if hasattr(results, "omni_normtest"):
        diagnostics_table += f"- **Omnibus**: {results.omni_normtest[0]:.3f}\n"
    if hasattr(results, "jarque_bera"):
        diagnostics_table += f"- **Jarque-Bera (JB)**: {results.jarque_bera[0]:.3f}\n"
    if hasattr(results, "skew"):
        diagnostics_table += f"- **Skew**: {results.skew:.3f}\n"
    if hasattr(results, "kurtosis"):
        diagnostics_table += f"- **Kurtosis**: {results.kurtosis:.3f}\n"
    if hasattr(results, "condition_number"):
        diagnostics_table += f"- **Condition No.**: {results.condition_number:.2e}\n"

    # Combine all parts into Markdown format
    markdown_output = header + stats_table + "\n### **Coefficient Table**\n\n" + coef_table + diagnostics_table

    return markdown_output

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
def dual_axis_histogram_line_chart(histogram_data, line_chart_data, x_label, y1_label, y2_label, title):
    """
    Draws a dual-axis chart:
    - Histogram (bar chart) from `histogram_data` (processed dictionary).
    - Line chart from `line_chart_data` (processed dictionary).
    
    Parameters:
        histogram_data (dict): Dictionary where keys are categories and values are counts.
        line_chart_data (dict): Dictionary where keys are the same as `histogram_data` but values are averages.
        x_label (str): Label for x-axis.
        y1_label (str): Label for histogram y-axis.
        y2_label (str): Label for line chart y-axis.
        title (str): Plot title.

    Example Usage:
        dual_axis_histogram_line_chart(demo_stats['Gene'], avg_age_per_defect, "Gene Defects Count", "Patient Count", "Average Age at dx", "Gene Defects vs. Age at Diagnosis")
    """
    # Convert dictionary data to sorted lists
    x_values = sorted(histogram_data.keys())  # Sorted defect count groups
    y1_values = [histogram_data[k] for k in x_values]  # Patient counts
    y2_values = [line_chart_data.get(k, None) for k in x_values]  # Avg Age at dx (match x_values)

    # Initialize figure
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Bar Chart (Histogram)
    color1 = "skyblue"
    ax1.bar(x_values, y1_values, color=color1, alpha=0.7, label=y1_label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Line Chart (Average Age)
    ax2 = ax1.twinx()
    color2 = "red"
    ax2.plot(x_values, y2_values, color=color2, marker="o", linestyle="-", linewidth=2, label=y2_label)
    ax2.set_ylabel(y2_label, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Titles and Legends
    plt.title(title)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Show plot
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