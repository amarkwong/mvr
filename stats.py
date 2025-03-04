import pandas as pd
import statsmodels.api as sm
from data import clean_data_for_ols_by_column
from ui import ols_to_markdown

def multivariate_linear_regression(df, x_columns, y_column):
    """
    Perform a multivariate linear regression on the given DataFrame.

    Parameters:
      df : pandas.DataFrame
          The dataset.
      x_columns : list of str
          The column names to be used as independent variables.
      y_column : str
          The column name to be used as the dependent variable.

    Prints the regression summary.
    """
    # Extract independent (X) and dependent (y) variables.
    clean_df=clean_data_for_ols_by_column(df,x_columns)

    # need to check y column as well
    clean_df=clean_data_for_ols_by_column(clean_df,[y_column])

    X = clean_df[x_columns]
    y = clean_df[y_column]
    
    # Add a constant term (intercept) to the independent variables.
    X = sm.add_constant(X)

    # Fit the OLS regression model.
    model = sm.OLS(y, X).fit()

    markdown_result = ols_to_markdown(model)

    # Print the Markdown
    print(markdown_result)

    # Save to file
    with open("ols_results.md", "w") as f:
        f.write(markdown_result)

    # Print the regression summary.
    print(model.summary())
def baseline_demographic(df, config):
    """
    Computes baseline demographic statistics based on `config["stats"]["baseline_demographic"]`.

    Parameters:
        df (pd.DataFrame): The dataset.
        config (dict): Loaded JSON config containing demographic stats.

    Returns:
        dict: A dictionary of computed statistics (only for 'table' mode).
    """
    result = {}

    # Get demographic configurations from the config file
    demographics = config.get("stats", {}).get("baseline_demographic", [])

    for demographic in demographics:
        if not demographic.get("enabled", True):
            continue  # Skip disabled settings

        display_mode = demographic.get("display_mode", "table")

        if display_mode == "table":
            # Table mode: Compute and store stats
            stats_list = demographic.get("stats", [])  # Array of stats for this demographic
            for stat_entry in stats_list:
                variable = stat_entry.get("name")
                metric = stat_entry.get("stats")

                if variable not in df.columns:
                    print(f"⚠️ Warning: Column '{variable}' not found in DataFrame. Skipping.")
                    continue

                if metric.lower() == "histogram":
                    result[variable] = {"histogram": df[variable].value_counts(dropna=False).to_dict()}

                elif metric.lower() == "median":
                    result[variable] = {"median": df[variable].median()}

                else:
                    print(f"⚠️ Warning: Unsupported metric '{metric}' for '{variable}'")

    return result  # Only contains stats for table mode