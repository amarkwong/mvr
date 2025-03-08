import pandas as pd
import statsmodels.api as sm
from data import data_fitting
from lifelines import CoxPHFitter, KaplanMeierFitter
from ui import ols_to_markdown
import matplotlib.pyplot as plt
import json

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
    clean_df=data_fitting(df,x_columns)

    # need to check y column as well
    clean_df=data_fitting(clean_df,[y_column])

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

import json
import pandas as pd
from lifelines import KaplanMeierFitter

def km_estimate(df, config_path="config.json"):
    """
    Computes Kaplan-Meier survival estimates based on settings in config.json.

    Parameters:
        df (pd.DataFrame): The dataset.
        config_path (str): Path to the configuration file.

    Returns:
        dict: Dictionary containing fitted KM models and survival data for each group.
    """
    # ✅ Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # ✅ Extract KM settings
    km_configs = config.get("stats", {}).get("km_estimate", [])

    if not isinstance(km_configs, list):
        print("⚠️ km_estimate in config.json should be a list. Skipping KM analysis.")
        return {}

    km_results = {}

    for km_config in km_configs:
        time_column = km_config.get("time_column", "Dx OS")
        event_column = km_config.get("event_column", "Death")
        group_column = km_config.get("group_column", None)
        enabled = km_config.get("enabled", True)

        if not enabled:
            print(f"⚠️ Skipping KM estimate for {group_column} (disabled in config).")
            continue

        if not all(col in df.columns for col in [time_column, event_column]):
            print(f"⚠️ Missing required columns for KM estimate: {[time_column, event_column]}. Skipping.")
            continue

        # ✅ Drop NaNs (Kaplan-Meier does not support missing values)
        km_df = df.dropna(subset=[time_column, event_column])
        km_df[event_column] = km_df[event_column].astype(int)  # Ensure event column is an integer

        print('df before filter',km_df)

        filtered_df = km_df[(km_df["BM Iron stores Class"] == 0) & (km_df["Death"] == 1)][["UR"]]

        print('filtered df',filtered_df)

        # ✅ Initialize KM model
        kmf = KaplanMeierFitter()

        km_data = {}

        if group_column and group_column in km_df.columns:
            # ✅ Convert column to integer if necessary
            km_df[group_column] = pd.to_numeric(km_df[group_column], errors="coerce").fillna(0).astype(int)

            unique_groups = sorted(km_df[group_column].unique())  # ✅ Ensure correct ordering

            if len(unique_groups) < 2:
                print(f"⚠️ Warning: Only one group found in {group_column}. Skipping stratified KM plot.")
                continue  # Skip plotting if there's only one group

            # ✅ Extract custom group labels from config (if available)
            group_labels = {
                str(item["value"]): item["label"]
                for item in km_config.get("group_label", [])
            }
            print(f"🔹 Loaded group labels: {group_labels}")  # Debugging output

            # ✅ Loop over groups
            for group in unique_groups:
                group_str = str(group)  # Convert group to string for lookup
                group_df = km_df[km_df[group_column] == group].copy()

                # ✅ Fetch custom label from config or fallback
                unique_label = group_labels.get(group_str, f"{group_column}: {group}")

                print(f"\n🧐 Processing group: {group} (n={len(group_df)}) - Assigned Label: {unique_label}")

                # ✅ Fit the KM model with correct label
                kmf = KaplanMeierFitter()
                kmf.fit(group_df[time_column], event_observed=group_df[event_column], label=unique_label)
                km_data[unique_label] = kmf  # ✅ Store using the correct label

        else:
            # ✅ Overall KM analysis without stratification
            kmf.fit(km_df[time_column], event_observed=km_df[event_column])
            km_data["Overall"] = kmf  # Store fitted KM model

        km_results[group_column if group_column else "Overall"] = km_data  # Store results

    print("\n📊 Final KM Results:")
    print(km_results)  # ✅ Debug output to verify structure

    return km_results

def cox_regression(df, config_path="config.json"):
    """
    Runs a Cox Proportional Hazards model based on settings in config.json.

    Parameters:
        df (pd.DataFrame): The dataset.
        config_path (str): Path to the configuration file.

    Returns:
        CoxPHFitter: The fitted Cox model.
    """
    # ✅ Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # ✅ Extract Cox settings from config.json
    cox_config = config.get("stats", {}).get("cox_regression", {})
    time_column = cox_config.get("time_column", "Dx OS")
    event_column = cox_config.get("event_column", "Death")
    independent_variables = cox_config.get("independent_variables", [])

    if not all(col in df.columns for col in [time_column, event_column] + independent_variables):
        missing_cols = [col for col in [time_column, event_column] + independent_variables if col not in df.columns]
        print(f"⚠️ Missing columns in dataset: {missing_cols}. Skipping Cox Regression.")
        return None

    # ✅ Prepare dataset (drop missing values)
    cox_df = df[[time_column, event_column] + independent_variables].dropna()

    # ✅ Convert categorical variables to dummy variables
    categorical_cols = [col for col in independent_variables if df[col].dtype == "object"]
    cox_df = pd.get_dummies(cox_df, columns=categorical_cols, drop_first=True)

    # ✅ Fit Cox Model
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col=time_column, event_col=event_column)

    # ✅ Print summary results
    print("\n📊 Cox Regression Summary:")
    cph.print_summary()

    # ✅ Plot hazard ratios
    cph.plot()
    
    return cph  # Return the fitted model for further analysis