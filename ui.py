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

def fetch_boxplot_data(demo_stats, category_name, numeric_name, metadata_lookup=None):
    """
    Extracts and formats box plot data (max, median, min) from `demo_stats` for visualization.

    Parameters:
        demo_stats (dict): Precomputed statistics from `baseline_demographic()`.
        category_name (str): The categorical variable (e.g., "Gender", "Gene Count").
        numeric_name (str): The numeric variable (e.g., "Age at dx").
        metadata_lookup (dict, optional): Mapping dictionary for converting numeric labels to readable names.

    Returns:
        pd.DataFrame: A formatted DataFrame with the necessary statistics for box plotting.
    """
    if numeric_name not in demo_stats:
        raise KeyError(f"{numeric_name} not found in demo_stats.")

    boxplot_stats = demo_stats[numeric_name]

    # Extracting overall statistics
    global_max = boxplot_stats.get("max", None)
    global_median = boxplot_stats.get("median", None)
    global_min = boxplot_stats.get("min", None)

    # Extracting per-category statistics
    per_category_max = boxplot_stats.get(f"max per {category_name}", {})
    per_category_median = boxplot_stats.get(f"median per {category_name}", {})
    per_category_min = boxplot_stats.get(f"min per {category_name}", {})

    # Convert numeric categories to readable labels if metadata_lookup is provided
    category_labels = list(per_category_median.keys())
    if metadata_lookup and category_name in metadata_lookup:
        category_labels = [metadata_lookup[category_name].get(k, str(k)) for k in category_labels]

    # Creating a structured DataFrame
    boxplot_data = pd.DataFrame({
        category_name: category_labels,  # Converted to readable labels
        "max": [per_category_max.get(k, global_max) for k in per_category_median.keys()],
        "median": [per_category_median.get(k, global_median) for k in per_category_median.keys()],
        "min": [per_category_min.get(k, global_min) for k in per_category_median.keys()]
    })

    boxplot_data_long = boxplot_data.melt(id_vars=[category_name], var_name="Statistic", value_name=numeric_name)

    return boxplot_data_long

def dual_axis_histogram_box_chart(histogram_data, df, x_label, y1_label, y2_label, title, color_config=None, box_opacity=0.5, box_width=0.5):
    """
    Draws a dual-axis chart:
    - Histogram (bar chart) from `histogram_data`.
    - Box plot from `df`, with customizable opacity and width.

    Parameters:
        histogram_data (dict): Precomputed histogram data (keys = categories, values = counts).
        df (pd.DataFrame): The full dataset with categorical and numeric columns.
        x_label (str): Label for x-axis.
        y1_label (str): Label for histogram y-axis.
        y2_label (str): Label for box plot y-axis.
        title (str): Plot title.
        color_config (dict, optional): Dictionary mapping x-axis categories to colors.
        box_opacity (float, optional): Opacity level for the box plot (default = 0.5).
        box_width (float, optional): Width of the box plot elements (default = 0.5).

    Example Usage:
        dual_axis_histogram_box_chart(demo_stats['Gender']['histogram'], df, "Gender", "Patient Count", "Age at dx", "Gender Distribution", box_opacity=0.5, box_width=0.4)
    """
    # Convert dictionary data to sorted lists
    x_values = sorted(histogram_data.keys())  # X-axis categories
    y1_values = [histogram_data[k] for k in x_values]  # Histogram counts

    # Assign colors based on config (default to gray)
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
    boxplot = sns.boxplot(
        x=df[x_label], 
        y=df[y2_label], 
        ax=ax2, 
        width=box_width,  # ✅ Custom box width
        patch_artist=True  # Allows us to modify box colors
    )

    # Set Box Plot Opacity
    for patch in boxplot.artists:
        patch.set_alpha(box_opacity)  # ✅ Adjust opacity of the box plot

    ax2.set_ylabel(y2_label, color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Titles and Legends
    plt.title(title)
    ax1.legend(loc="upper left")

    # Show plot
    plt.show()



def display_demographic_data(config, df, category_column, numeric_column, demo_stats):
    """
    Displays demographic data as a Markdown table.

    Parameters:
        config (dict): Configuration loaded from config.json.
        df (pd.DataFrame): The full dataset.
        category_column (str): The categorical variable for the x-axis (e.g., "Gender", "Gene Count").
        numeric_column (str): The numeric variable for the box plot (e.g., "Age at dx").
        demo_stats (dict): Precomputed statistics from `baseline_demographic()`.

    Returns:
        None
    """
    # Extract stats display mode from config
    display_mode = "table"  # Default
    for item in config["stats"].get("baseline_demographic", []):
        if item.get("name") == category_column:
            display_mode = item.get("display_mode", "table")

    if display_mode == "table":
        print(f"\n📊 **Displaying {category_column} Demographic as a Table:**")

        # Retrieve statistics for the given category
        category_stats = demo_stats.get(category_column, {})

        if category_stats:
            # Convert dictionary stats to DataFrame for markdown rendering
            stats_df = pd.DataFrame.from_dict(category_stats, orient="index", columns=[category_column])

            # ✅ Display DataFrame in Markdown format
            print(stats_df.to_markdown(index=True))  # Prints markdown-formatted table

        else:
            print(f"⚠️ Warning: No demographic data available for {category_column}")

    elif display_mode == "chart":
        print(f"⚠️ Chart mode selected. This function does not handle charts. Use `dual_axis_histogram_box_chart()`.")

    else:
        print(f"⚠️ Unknown display mode '{display_mode}' for {category_column}. Defaulting to table.")


# * Chart plotting
def plot_km_survival_curves(km_results):
    """
    Plots Kaplan-Meier survival curves from precomputed KM models.

    Parameters:
        km_results (dict): Dictionary containing fitted Kaplan-Meier models for each group.

    Returns:
        None (displays plots).
    """
    for group_name, km_models in km_results.items():
        plt.figure(figsize=(8, 5))

        for subgroup, kmf in km_models.items():
            kmf.plot_survival_function(label=subgroup)

        # ✅ Formatting the plot
        plt.title(f"Kaplan-Meier Survival Curve ({group_name})")
        plt.xlabel("Time (Months)")
        plt.ylabel("Survival Probability")
        plt.legend(title=group_name if group_name != "Overall" else "Survival")
        plt.grid(True)
        plt.show()