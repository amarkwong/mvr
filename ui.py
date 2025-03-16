import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import sys
import re
import io
import base64
import statsmodels.api as sm
import seaborn as sns
from colorama import init, Fore, Style
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator

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

def get_report_path(config_path="config.json"):
    """
    Reads the report file path from the config file.
    
    Parameters:
        config_path (str): Path to the config JSON file.
        
    Returns:
        str: The report file path.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["file"]["report_path"]

def add_plot_to_report(figure_path, alt_text="Plot", config_path="config.json"):
    """
    Appends a markdown image link to the report file.

    Parameters:
        figure_path (str): Path or URL to the plot image.
        alt_text (str): Alternative text for the image.
        config_path (str): Path to the config JSON file.
    """
    report_path = get_report_path(config_path)
    markdown_image = f"![{alt_text}]({figure_path})\n\n"
    with open(report_path, "a") as f:
        f.write(markdown_image)
    print(f"Plot added to report at {report_path}")

def add_table_to_report(table, config_path="config.json"):
    """
    Appends a markdown-formatted table to the report file.
    If table is a pandas DataFrame, it is converted to markdown format.
    
    Parameters:
        table (pd.DataFrame or str): Table to be added. If not a DataFrame, its string 
                                     representation will be used.
        config_path (str): Path to the config JSON file.
    """
    report_path = get_report_path(config_path)
    if isinstance(table, pd.DataFrame):
        markdown_table = table.to_markdown(index=False)
    else:
        markdown_table = str(table)
    markdown_table += "\n\n"
    with open(report_path, "a") as f:
        f.write(markdown_table)
    print(f"Table added to report at {report_path}")

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
    headers = ["Variable", "Coef", "Std Err", "t", "P&gt;\\|t\\|", "[0.025", "0.975]"]

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

def draw_bar_chart_from_series(ax, series, metadata_lookup=None, color_config=None, config_path="config.json"):
    """
    Draws a bar chart on the provided axis using a DataFrame column (pandas Series).
    
    The function computes the histogram of the Series, maps the raw values to descriptive
    labels using the provided metadata_lookup, then fetches corresponding colors from the
    color configuration (from config.json under ui.category.color). If a descriptive label
    is not found in the color configuration, the default color is used.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis on which to draw the bar chart.
        series (pd.Series): The DataFrame column (e.g. df['Gender']) on which to run the histogram.
        metadata_lookup (dict, optional): Mapping for the series name to convert raw values to labels.
            Example: {"Gender": {1: "male", 0: "female"}}
        color_config (dict, optional): Mapping of descriptive labels to colors.
            Example: {"male": "#457b9d", "female": "#e63946", "default": "#83c5be"}
            If not provided, it is loaded from config.json.
        config_path (str): Path to the configuration JSON file.
    
    Returns:
        None
    """
    # Load color configuration from config.json if not provided
    if color_config is None:
        with open(config_path, "r") as f:
            config = json.load(f)
        color_config = config.get("ui", {}).get("category", {}).get("color", {})
    
    # Compute the histogram from the Series (including NaNs if desired)
    hist = series.value_counts(dropna=False)
    
    # Try to sort the histogram keys numerically; if not possible, sort alphabetically.
    try:
        sorted_keys = sorted(hist.index, key=lambda x: float(x))
    except (ValueError, TypeError):
        sorted_keys = sorted(hist.index, key=lambda x: str(x))
    
    # If metadata_lookup is provided and contains a mapping for this column,
    # convert the raw keys to descriptive labels.
    if metadata_lookup and series.name in metadata_lookup:
        mapping = metadata_lookup[series.name]
        mapped_labels = []
        for k in sorted_keys:
            try:
                # Convert the raw key to a numeric type for lookup
                numeric_key = int(float(k))
            except (ValueError, TypeError):
                numeric_key = k
            label = mapping.get(numeric_key, k)
            mapped_labels.append(label)
    else:
        mapped_labels = [str(k) for k in sorted_keys]
    
    # Get histogram counts in the order of sorted_keys
    y_values = [hist[k] for k in sorted_keys]
    
    # Determine colors for bars using the mapped descriptive labels.
    default_color = color_config.get("default", "#AAAAAA")
    bar_colors = [color_config.get(str(label).lower(), default_color) for label in mapped_labels]
    
    # Plot the bar chart using the mapped labels and corresponding counts/colors.
    ax.bar(mapped_labels, y_values, color=bar_colors, alpha=0.7)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Count")
    ax.tick_params(axis="y", labelcolor="black")

def add_opacity_to_hex(hex_color, alpha):
    """
    Converts a hex color (e.g. "#457b9d") to an 8-digit hex including the desired alpha.
    """
    rgba = mcolors.to_rgba(hex_color, alpha=alpha)
    return mcolors.to_hex(rgba, keep_alpha=True)

def draw_boxplot(ax, df, x_label, y_label, metadata_lookup=None, box_width=0.5, box_opacity=0.5,
                 box_color_config=None, config_path="config.json"):
    """
    Draws a box plot on the provided axis.
    
    This function:
      - Loads the boxplot color configuration from config.json (under ui.boxplot.color)
        if not provided.
      - Overrides box_opacity (and optionally box_width) using values from the config.
      - Computes the sorted unique raw values from the x-axis column.
      - Uses metadata_lookup to map these raw values to descriptive labels.
      - Converts each hex color from the config to an 8-digit hex (RGBA) with the desired opacity.
      - Constructs a seaborn palette from these RGBA hex codes.
      - Uses the hue parameter (with legend disabled) to avoid the deprecation warning.
      - Sets a fixed locator for tick labels to avoid warnings.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis on which to draw the boxplot.
        df (pd.DataFrame): DataFrame containing the data.
        x_label (str): Column name for the x-axis.
        y_label (str): Column name for the y-axis.
        box_width (float): Width of the box plot elements.
        box_opacity (float): Fallback opacity for the box plot if not specified in config.
        box_color_config (dict, optional): Mapping of category labels to colors.
            Example: {"male": "#118ab2", "female": "#ef476f", "default": "#06d6a0", "opacity": 0.2}
        metadata_lookup (dict, optional): Mapping for x_label to convert raw values to descriptive labels.
            Example: {"Gender": {1: "male", 0: "female"}}
        config_path (str): Path to the configuration JSON file.
    """
    # Load box_color_config from config.json if not provided.
    if box_color_config is None:
        with open(config_path, "r") as f:
            config = json.load(f)
        box_color_config = config.get("ui", {}).get("boxplot", {}).get("color", {})
        box_width = config.get("ui", {}).get("boxplot", {}).get("width", box_width)
    
    # Override box_opacity from config if available.
    box_opacity = box_color_config.get("opacity", box_opacity)
    
    # Compute unique raw values from df[x_label].
    raw_values = df[x_label].dropna().unique()
    try:
        sorted_raw = sorted(raw_values, key=lambda x: float(x))
    except Exception:
        sorted_raw = sorted(raw_values, key=lambda x: str(x))
    
    # Map raw values to descriptive labels using metadata_lookup if available.
    if metadata_lookup and x_label in metadata_lookup:
        mapping = metadata_lookup[x_label]
        mapped_labels = []
        for v in sorted_raw:
            try:
                numeric_val = int(float(v))
            except (ValueError, TypeError):
                numeric_val = v
            mapped_labels.append(mapping.get(numeric_val, v))
    else:
        mapped_labels = [str(v) for v in sorted_raw]
    
    # Build a list of colors for each mapped label.
    default_color = box_color_config.get("default", "#06d6a0")
    palette_colors = [box_color_config.get(str(label).lower(), default_color) for label in mapped_labels]
    
    # Apply opacity to each hex color (baking it into an 8-digit hex code).
    palette_colors_rgba = [add_opacity_to_hex(color, box_opacity) for color in palette_colors]
    
    # Create a seaborn palette from these RGBA hex codes.
    palette = sns.color_palette(palette_colors_rgba)
    
    # Convert x values to string for categorical processing.
    df_x = df[x_label].astype(str)
    sorted_raw_str = [str(s) for s in sorted_raw]
    
    # Draw the boxplot using hue to avoid the deprecation warning.
    # Setting dodge=False and legend=False ensures a single box per category.
    boxplot = sns.boxplot(
        x=df_x, y=df[y_label], ax=ax,
        width=box_width, order=sorted_raw_str,
        palette=palette,
        hue=df_x, dodge=False, legend=False
    )
        # Force the x-axis to have as many ticks as mapped_labels
    ax.set_xticks(range(len(mapped_labels)))
    ax.set_xticklabels(mapped_labels)
    
    # Set fixed tick locations to avoid warnings.
    # ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
    # # Update tick labels to the descriptive labels.
    # ax.set_xticklabels(mapped_labels)
    
    ax.set_ylabel(y_label, color="red")
    ax.tick_params(axis="y", labelcolor="red")

def dual_axis_histogram_box_chart(histogram_data, df, x_label, y1_label, y2_label, title,
                                  metadata_lookup=None,
                                  color_config=None, box_color_config=None,
                                  box_opacity=0.5, box_width=0.5,
                                  report_path=None, config_path="config.json"):
    """
    Combines a bar chart and a box plot into a dual-axis chart.
    The bar chart is drawn using draw_bar_chart and the box plot using draw_boxplot.
    The final plot is saved as an inline base64 image to a Markdown report file if report_path is provided.

    Parameters:
        histogram_data (dict): Histogram data (keys = categories, values = counts).
        df (pd.DataFrame): DataFrame containing the data.
        x_label (str): Column name for x-axis.
        y1_label (str): Label for the histogram's y-axis.
        y2_label (str): Label for the boxplot's y-axis.
        title (str): Plot title.
        metadata_lookup (dict, optional): Mapping for x_label to convert raw keys to labels.
        color_config (dict, optional): Mapping of labels to colors for the bar chart.
        box_color_config (dict, optional): Mapping of labels to colors for the box plot.
        box_opacity (float, optional): Opacity for the box plot.
        box_width (float, optional): Width for the box plot.
        report_path (str, optional): Markdown report file path.
        config_path (str): Path to configuration JSON.
    """
    # Retrieve report_path from config if not provided.
    if report_path is None:
        with open(config_path, "r") as f:
            config = json.load(f)
        report_path = config.get("file", {}).get("report_path", None)
    
    # Create figure and primary axis.
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Draw bar chart on ax1.
    draw_bar_chart(ax1, histogram_data, x_label, metadata_lookup, color_config)
    
    # Create secondary axis for boxplot.
    ax2 = ax1.twinx()
    draw_boxplot(ax2, df, x_label, y2_label, box_width, box_opacity, box_color_config)
    
    # Set overall title and legend.
    plt.title(title)
    ax1.legend(loc="upper left")
    
    # Save the combined figure to an in-memory buffer as PNG.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_data = buf.read()
    buf.close()
    plt.close(fig)
    
    # Encode image in base64 for Markdown embedding.
    img_base64 = base64.b64encode(img_data).decode("utf-8")
    markdown_image = f"![Dual Axis Chart](data:image/png;base64,{img_base64})\n\n"
    
    if report_path:
        with open(report_path, "a") as f:
            f.write(markdown_image)
        print(f"Dual axis chart added to report at {report_path}")
    else:
        print(markdown_image)

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
def plot_km_survival_curves(km_results, report_path=None, config_path="config.json"):
    """
    Plots Kaplan–Meier survival curves with customizable colors, legends, and captions.
    For each group, the function generates a plot, encodes it as an inline base64 PNG image,
    and appends the corresponding markdown to the report file if provided.

    Parameters:
        km_results (dict): Dictionary containing fitted Kaplan–Meier models for each group.
        report_path (str, optional): Markdown report file path to which inline images will be appended.
            If not provided, it is read from the configuration file.
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Use report_path from config if not explicitly provided
    if report_path is None:
        report_path = config.get("file", {}).get("report_path", None)

    # Extract KM plot configuration
    km_config = config.get("ui", {}).get("km_plot", {})
    show_ci = km_config.get("show_confidence_interval", True)
    show_legend = km_config.get("show_legend", True)
    captions = km_config.get("captions", {})
    palette = km_config.get("palette", {})

    # Process each group in km_results
    for group_name, km_models in km_results.items():
        # Create a new figure for the group
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot survival function for each subgroup
        for subgroup, kmf in km_models.items():
            color = palette.get(subgroup, palette.get("default", "#000000"))
            kmf.plot_survival_function(ax=ax, label=subgroup, ci_show=show_ci, color=color)
        
        # Format the plot
        ax.set_title(f"Kaplan–Meier Survival Curve ({group_name})")
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel("Survival Probability")
        if show_legend:
            ax.legend(title=group_name if group_name != "Overall" else "Survival")
        else:
            ax.legend().remove()
        ax.grid(True)
        
        # Add caption if available
        caption = captions.get(group_name, "")
        if caption:
            plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=10)
        
        # Save the plot to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_data = buf.read()
        buf.close()
        plt.close(fig)
        
        # Encode image to base64 and create the markdown image syntax
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        markdown_image = f"![KM Survival Curve ({group_name})](data:image/png;base64,{img_base64})\n\n"
        
        # Append the markdown image to the report file if a report_path is available
        if report_path:
            with open(report_path, "a") as f:
                f.write(markdown_image)
            print(f"KM Survival curve for {group_name} added to report at {report_path}")
        else:
            # Otherwise, simply print the markdown text
            print(markdown_image)

def plot_cox_model(cph, report_path=None, config_path="config.json"):
    """
    Plots the hazard ratios with a log-scaled x-axis, embeds the Cox model summary and
    an inline base64-encoded image of the plot into a Markdown report file.
    
    Parameters:
        cph (CoxPHFitter): The fitted Cox model.
        report_path (str, optional): The Markdown report file path. If not provided,
                                     it will be read from config.json.
        config_path (str): Path to the config JSON file.
    """
    if cph is None:
        print("No Cox model to plot.")
        return

    # Use the report_path from config if not provided
    if report_path is None:
        report_path = get_report_path(config_path)

    # Create Markdown summary text
    markdown_summary = "\n📊 Cox Regression Summary:\n" + cph.summary.to_markdown() + "\n\n"

    # Append the summary to the report file if provided; otherwise, print it.
    if report_path:
        with open(report_path, "a") as f:
            f.write(markdown_summary)
        print(f"Cox summary added to report at {report_path}")
    else:
        print(markdown_summary)

    # Extract hazard ratios and confidence intervals from the model summary
    hr_exp = cph.summary["exp(coef)"]
    ci_lower = cph.summary["exp(coef) lower 95%"]
    ci_upper = cph.summary["exp(coef) upper 95%"]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        hr_exp, hr_exp.index,
        xerr=[hr_exp - ci_lower, ci_upper - hr_exp],
        fmt="o", color="black", ecolor="red", capsize=5
    )
    ax.set_xscale("log")
    ax.axvline(1, color="gray", linestyle="--")  # Reference line at HR = 1
    ax.set_xlabel("Hazard Ratio (log scale)")
    ax.set_ylabel("Features")
    ax.set_title("Cox Proportional Hazards Model - Log Scale")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_data = buf.read()
    buf.close()
    plt.close(fig)

    # Encode image in base64 for embedding
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    markdown_image = f"![Cox Plot](data:image/png;base64,{img_base64})\n\n"

    # Append the inline image markdown to the report file if provided; otherwise, print it.
    if report_path:
        with open(report_path, "a") as f:
            f.write(markdown_image)
        print(f"Inline plot image added to report at {report_path}")
    else:
        print(markdown_image)