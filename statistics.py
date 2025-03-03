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
    Compute baseline demographics for the given DataFrame.

    Parameters:
      df : pandas.DataFrame
          The dataset.
      baseline_vars : list of str
          Baseline variable names. A leading '*' (e.g., '*Gender') is removed.
      baseline_metric : list of str
          Metrics for each variable. Supported metrics are:
          - "histogram": For categorical variables, compute value_counts.
          - "median": For numeric variables, compute the median.
          - "count histogram": For list-type columns (e.g., Gene),
            compute a histogram of the list lengths.
    
    Returns:
      dict
          A dictionary where keys are cleaned variable names and values are the computed statistics.
    """
    result = {}
    
    for var, metric in zip(baseline_vars, baseline_metric):
        # Clean the variable name by removing any leading '*' and extra spaces.
        clean_var = var.lstrip('*').strip()
        
        if clean_var not in df.columns:
            print(f"Warning: Column '{clean_var}' not found in DataFrame.")
            result[clean_var] = None
            continue
        
        # Process based on the metric.
        if metric.lower() == 'histogram':
            hist = df[clean_var].value_counts(dropna=False).to_dict()
            result[clean_var] = hist
        
        elif metric.lower() == 'median':
            median_val = df[clean_var].median()
            result[clean_var] = median_val
        
        elif metric.lower() == 'count histogram':
            # For a column where each value is a list (e.g., Gene), compute the count histogram of list lengths.
            lengths = df[clean_var].apply(lambda x: len(x) if isinstance(x, list) else 0)
            count_hist = lengths.value_counts().sort_index().to_dict()
            result[clean_var] = count_hist
        
        else:
            print(f"Warning: Unknown metric '{metric}' for column '{clean_var}'.")
            result[clean_var] = None

    return result