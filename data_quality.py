import os
import json
import pandas as pd

def clean_data_for_ols_by_column(df, columns_to_check, config_path="conf/data_fitting.json"):
    """
    For each column in columns_to_check, check for missing values.
    For any column with missing data, print a message listing the UR values missing data.
    Then, either use a preset fitting option from the configuration file or prompt the user.
    Allowed options are:
       'drop'   : drop rows where the column is missing,
       'mean'   : fill with the mean (numeric columns only),
       'medium' : fill with the median (numeric columns only),
       'mode'   : fill with the mode,
       'zero'   : fill with 0,
       'calc': calculate the value using another column.
                    In this case, the configuration should be a dictionary, for example:
                    {
                      "first_input": "Date last FollowUp (or death)",
                      "operator": "-",
                      "second_input": "Date AML dx",
                      "unit": "month"
                    }
    If a setting is not present in the config file, the user is prompted and the choice is saved.
    
    Returns:
       The DataFrame with missing values in the specified columns handled.
    """
    allowed_options = ['drop', 'mean', 'medium', 'mode', 'zero', 'calc']
    
    # Load existing configuration if it exists; otherwise, use an empty config.
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    for col in columns_to_check:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame, skipping.")
            continue
        
        # Get the list of patient IDs (UR values) where the column is missing.
        missing_urs = df.loc[df[col].isnull(), "UR"].tolist()
        if missing_urs:
            print(f"For column '{col}', the following UR values have missing data: {missing_urs}")
            # Look for a pre-configured option.
            if col in config:
                option = config[col]
                # If the config value is a dict, we treat it as the 'calculate' option.
                if isinstance(option, dict):
                    option_type = "calculate"
                else:
                    option_type = option
                print(f"Using configured option for column '{col}': {option}")
            else:
                print(f"Choose a method to handle missing data for column '{col}':")
                print(" Options: 'drop', 'mean', 'medium', 'mode', 'zero', 'calc'")
                option = input("Enter your choice: ").strip().lower()
                while option not in allowed_options:
                    option = input("Invalid input. Please enter one of ['drop','mean','medium','mode','zero','calc']: ").strip().lower()
                if option == "calculate":
                    # Prompt for calculation details.
                    print(f"Enter calculation details for column '{col}':")
                    first_input = input("Enter the column name for first input (e.g., Date last FollowUp (or death)): ").strip()
                    operator = input("Enter the operator (only '-' is supported): ").strip()
                    second_input = input("Enter the column name for second input (e.g., Date AML dx): ").strip()
                    unit = input("Enter the unit (e.g., 'month'): ").strip()
                    option = {
                        "first_input": first_input,
                        "operator": operator,
                        "second_input": second_input,
                        "unit": unit
                    }
                # Save the chosen option.
                config[col] = option
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
            
            # Apply the chosen option.
            # Handle the 'calculate' option:
            if (isinstance(option, dict)) or (option == "calc"):
                calc_conf = option if isinstance(option, dict) else None
                if calc_conf is None:
                    print(f"Calculation configuration for column '{col}' is missing, skipping calculation.")
                else:
                    # Extract inputs from the configuration.
                    first_input = calc_conf["first_input"]
                    second_input = calc_conf["second_input"]
                    op = calc_conf["operator"]
                    unit = calc_conf["unit"].lower()
                    # We support only subtraction ('-') and unit 'month' in this example.
                    if op != "-" or unit != "month":
                        print(f"Calculation for '{col}' only supports '-' and unit 'month'. Skipping calculation.")
                    else:
                        # Convert the specified columns to datetime.
                        df[first_input] = pd.to_datetime(df[first_input], errors="coerce")
                        df[second_input] = pd.to_datetime(df[second_input], errors="coerce")
                        
                        def calculate_value(row):
                            if pd.isnull(row[col]):
                                if pd.notnull(row[first_input]) and pd.notnull(row[second_input]):
                                    delta = row[first_input] - row[second_input]
                                    # Convert the difference to months (approximate conversion: 30 days per month)
                                    months = delta.days / 30
                                    return months
                            return row[col]
                        
                        df[col] = df.apply(calculate_value, axis=1)
                        print(f"Missing '{col}' values have been calculated using: {first_input} {op} {second_input} in {unit}s.")
            elif option == "drop":
                df = df.dropna(subset=[col])
                print(f"Rows with missing '{col}' have been dropped.")
            elif option == "mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                    print(f"Missing '{col}' values have been filled with the mean.")
                else:
                    print(f"Column '{col}' is not numeric; skipping mean imputation.")
            elif option == "medium":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                    print(f"Missing '{col}' values have been filled with the median.")
                else:
                    print(f"Column '{col}' is not numeric; skipping median imputation.")
            elif option == "mode":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                    print(f"Missing '{col}' values have been filled with the mode.")
                else:
                    print(f"Column '{col}' has no mode; leaving missing values.")
            elif option == "zero":
                df[col] = df[col].fillna(0)
                print(f"Missing '{col}' values have been filled with zero.")
        else:
            print(f"No missing data detected in column '{col}'.")
    
    return df