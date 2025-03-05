import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import re
import statsmodels.api as sm
import seaborn as sns
from colorama import init, Fore, Style

# Initialize colorama (needed for Windows)
init(autoreset=True)
# const
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

MARKDOWN_FILE = "analysis.md"

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

def dual_axis_histogram_box_chart(histogram_data, df, x_label, y1_label, y2_label, title, 
                                  color_config=None, box_opacity=0.5, box_width=0.5, save_file=None):
    """
    Draws a dual-axis chart:
    - Histogram (bar chart) from `histogram_data`.
    - Box plot from `df`, with customizable opacity and width.
    - Saves to file if `save_file` is provided.

    Parameters:
        save_file (str, optional): If provided, saves the chart instead of displaying it.
    """
    # Convert dictionary data to sorted lists
    x_values = sorted(histogram_data.keys())  
    y1_values = [histogram_data[k] for k in x_values]  

    # Assign colors based on config
    bar_colors = [color_config.get(cat, "#AAAAAA") for cat in x_values] if color_config else ["#AAAAAA"] * len(x_values)

    # Initialize figure
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Bar Chart (Histogram)
    ax1.bar(x_values, y1_values, color=bar_colors, alpha=0.7, label=y1_label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color="black")
    ax1.tick_params(axis='y', labelcolor="black")

    # Create second y-axis for Box Plot
    ax2 = ax1.twinx()

    # Draw Box Plot with Custom Opacity & Width
    boxplot = sns.boxplot(x=df[x_label], y=df[y2_label], ax=ax2, width=box_width, patch_artist=True)
    for patch in boxplot.artists:
        patch.set_alpha(box_opacity)

    ax2.set_ylabel(y2_label, color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    plt.title(title)
    ax1.legend(loc="upper left")

    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def generate_markdown_table(category_column, demo_stats, save_file=None, metadata_lookup=None):
    """
    Generates a Markdown-formatted table for demographic data.
    
    - Uses metadata mapping to replace numeric values with labels when applicable.
    - Formats category labels for improved readability.

    Parameters:
        category_column (str): The demographic category to process.
        demo_stats (dict): The precomputed statistics from `baseline_demographic()`.
        save_file (str, optional): If provided, writes the table to a Markdown file.
        metadata_lookup (dict, optional): Mapping dictionary to replace numeric values with labels.

    Returns:
        str: Markdown formatted table.
    """
    category_stats = demo_stats.get(category_column, {})

    if not category_stats:
        return f"⚠️ No demographic data available for {category_column}\n"

    # ✅ Handle nested dictionaries properly
    flat_data = {}
    for key, value in category_stats.items():
        if isinstance(value, dict):  # Handle histogram or grouped statistics
            for sub_key, sub_value in value.items():
                # ✅ Replace numeric keys with human-readable labels
                if metadata_lookup and category_column in metadata_lookup:
                    readable_label = metadata_lookup[category_column].get(sub_key, sub_key)
                else:
                    readable_label = f"{category_column} ({sub_key})"  # Fallback label
                
                flat_data[readable_label] = sub_value
        else:
            flat_data[key] = value

    # Convert flattened data to DataFrame
    stats_df = pd.DataFrame(list(flat_data.items()), columns=["Category", category_column])

    if stats_df.empty:
        return f"⚠️ No data available for {category_column}\n"

    markdown_table = stats_df.to_markdown(index=False)

    if save_file:
        with open(save_file, "a") as md:
            md.write(f"### {category_column} Statistics\n")
            md.write(markdown_table + "\n\n")

    return markdown_table

def display_demographic_data(config, df, category_column, numeric_column, demo_stats, save_file=None, metadata_lookup=None):
    """
    Displays demographic data as a Markdown table or a saved chart.

    - Calls `dual_axis_histogram_box_chart()` for charts if enabled.
    - Calls `generate_markdown_table()` for tables if enabled.
    - Uses metadata lookup for replacing numeric values with labels.

    Parameters:
        config (dict): Configuration loaded from config.json.
        df (pd.DataFrame): The full dataset.
        category_column (str): The categorical variable for the x-axis (e.g., "Gender", "Gene Count").
        numeric_column (str): The numeric variable for the box plot (e.g., "Age at dx").
        demo_stats (dict): Precomputed statistics from `baseline_demographic()`.
        save_file (str, optional): If provided, saves the output instead of displaying it.
        metadata_lookup (dict, optional): Used to map numeric values to labels.
    """

    demographic_config = next((item for item in config["stats"].get("baseline_demographic", []) if item.get("name") == category_column), None)

    if not demographic_config:
        print(f"⚠️ Warning: No configuration found for {category_column}")
        return

    display_mode = demographic_config.get("display_mode", "table")

    # ✅ Handle Charts
    if display_mode == "chart" and "histogram_variable" in demographic_config and "numeric_variable" in demographic_config:
        if category_column in demo_stats and "histogram" in demo_stats[category_column]:
            chart_filename = f"{category_column}_chart.png" if save_file else None

            dual_axis_histogram_box_chart(
                histogram_data=demo_stats[category_column]["histogram"],
                df=df,
                x_label=category_column,
                y1_label="Patient Count",
                y2_label=numeric_column,
                title=f"{category_column} vs. {numeric_column} Distribution",
                color_config=config["ui"]["histogram"]["color"],
                box_opacity=config["ui"]["boxplot"].get("opacity", 0.5),
                box_width=config["ui"]["boxplot"].get("width", 0.5),
                save_file=chart_filename
            )

            # ✅ Embed chart in Markdown
            if save_file:
                with open(save_file, "a") as md:
                    md.write(f"### {category_column} Distribution\n")
                    md.write(f"![{category_column} Chart]({chart_filename})\n\n")

    # ✅ Handle Tables
    if display_mode == "table":
        generate_markdown_table(category_column, demo_stats, save_file, metadata_lookup)