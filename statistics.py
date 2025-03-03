import pandas as pd
import statsmodels.api as sm
from data_quality import clean_data_for_ols_by_column
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

def baseline_demographic(df, baseline_vars, baseline_metric):
    """
    Compute baseline demographics for the given DataFrame, supporting multiple metrics per variable.

    Parameters:
      df : pandas.DataFrame
          The dataset.
      baseline_vars : list of str
          Baseline variable names.
      baseline_metric : list of str or list of lists
          Metrics for each variable. Supported metrics:
          - "histogram": Compute value_counts for categorical variables.
          - "median": Compute median of numeric variables.
          - "count histogram": Compute a histogram of list lengths (for Gene mutations).
          - "median per <Category>": Compute median grouped by another variable (e.g., per Gender).

    Returns:
      dict
          A dictionary where keys are variable names and values are the computed statistics.
    """
    result = {}

    for var, metrics in zip(baseline_vars, baseline_metric):
        # Clean the variable name by removing any leading '*' and extra spaces.
        clean_var = var.lstrip('*').strip()

        if clean_var not in df.columns:
            print(f"Warning: Column '{clean_var}' not found in DataFrame.")
            result[clean_var] = None
            continue

        # Convert metrics to list if not already (for single-metric cases)
        if not isinstance(metrics, list):
            metrics = [metrics]

        # Store results for this variable
        variable_results = {}

        for metric in metrics:
            if metric.lower() == 'histogram':
                variable_results["histogram"] = df[clean_var].value_counts(dropna=False).to_dict()

            elif metric.lower() == 'median':
                variable_results["median"] = df[clean_var].median()

            elif metric.lower() == 'count histogram':
                lengths = df[clean_var].apply(lambda x: len(x) if isinstance(x, list) else 0)
                variable_results["count histogram"] = lengths.value_counts().sort_index().to_dict()

            elif metric.lower() == 'median per gender' and 'Gender' in df.columns:
                variable_results["median per Gender"] = df.groupby("Gender")[clean_var].median().to_dict()

            elif metric.lower() == 'median per gene count' and 'Gene Count' in df.columns:
                variable_results["median per Gene Count"] = df.groupby("Gene Count")[clean_var].median().to_dict()

            else:
                print(f"Warning: Unknown or unsupported metric '{metric}' for column '{clean_var}'.")
        
        # Store the computed results
        result[clean_var] = variable_results

    return result