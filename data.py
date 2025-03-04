import pandas as pd
import os
import re
import json
from ui import styled_print

def process_headers(columns):
    header_names = []
    descriptions = {}
    for col in columns:
        col_str = str(col).strip()
        if '#' in col_str:
            actual, desc = col_str.split('#', 1)
            actual = actual.strip().lstrip('*').rstrip('?')
            desc = desc.strip()
            header_names.append(actual)
            descriptions[actual] = desc
        else:
            header_names.append(col_str.lstrip('*'))
    return header_names, descriptions

def combine_gene_info(row):
    """
    Combine gene-related columns into a nested list of dictionaries.
    Each gene record is taken from the corresponding entries in the aggregated lists.
    Skip any gene record where the gene value is missing or equals 0 (or "0").
    """
    gene_records = []
    genes = row["Gene"]
    vafs = row["VAF% G1"]
    tiers = row["Tier"]
    var_descs = row["Variant description"]
    for gene, vaf, tier, var_desc in zip(genes, vafs, tiers, var_descs):
        if pd.isna(gene) or gene == 0 or str(gene).strip() == "0":
            continue
        gene_records.append({
            "name": gene,
            "VAF% G1": vaf,
            "Tier": tier,
            "Variant description": var_desc
        })
    return gene_records

def generate_metadata_mapping(header_metadata):
    """
    Parses the header_metadata dictionary and creates a lookup mapping for categorical values.

    Parameters:
        header_metadata (dict): Dictionary containing metadata descriptions.

    Returns:
        dict: A nested dictionary where keys are column names, and values are dictionaries mapping numeric codes to labels.
    """
    mapping_dict = {}

    for column, mapping_str in header_metadata.items():
        value_mapping = {}
        matches = re.findall(r"(\d+)\s*[=:-]?\s*([\w\s]+)", mapping_str)  # Handles "1 = label" or "1 - label" formats
        
        for num, label in matches:
            if label.strip():  # Ensure the label is not empty
                value_mapping[int(num)] = label.strip().lower()  # Normalize labels

        if value_mapping:  # Only store mappings that are valid
            mapping_dict[column] = value_mapping

    return mapping_dict

def load_and_clean_data(file_path):
    """
    Load Excel data and aggregate rows by 'UR' so that non-gene columns remain as in the first row,
    and gene-related columns are collected into lists. Then combine the gene-related lists into a single
    nested "Gene" list.
    
    This function does NOT apply a global forward-fill, so that if a column like 'Ferritin' is null
    in the first row for a patient, it remains null.
    
    Returns:
      - aggregated: the cleaned, aggregated DataFrame.
      - header_metadata: dictionary of header descriptions.
    """
    na_values = ["NA", "na", "N/A", "n/a", "N/a"]
    df = pd.read_excel(file_path, engine="openpyxl", na_values=na_values)
    df["UR"] = df["UR"].ffill()
    # Process headers.
    raw_headers = list(df.columns)
    new_columns, header_metadata = process_headers(raw_headers)
    df.columns = new_columns

    # Sort by patient identifier (UR) to ensure rows for the same patient remain together.
    if "UR" in df.columns:
        df = df.sort_values("UR")
    
    # Define gene-related columns.
    gene_columns = ["Gene", "VAF% G1", "Tier", "Variant description"]
    # (No forward fill is applied here, so that each raw row remains intact.)
    
    # Build an aggregation dictionary.
    agg_dict = {}
    for col in df.columns:
        if col == "UR":
            continue  # Grouping key.
        if col in gene_columns:
            # Using lambda with a default parameter to capture the current col.
            agg_dict[col] = lambda x, col=col: x.tolist()
        else:
            agg_dict[col] = lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
    
    aggregated = df.groupby("UR", as_index=False).agg(agg_dict)
    
    # Combine gene-related columns into a single nested "Gene" list.
    aggregated["Gene"] = aggregated.apply(combine_gene_info, axis=1)
    aggregated = aggregated.drop(columns=["VAF% G1", "Tier", "Variant description"])
    
    aggregated.to_json("merged_output.json", orient="records", indent=2)
    
    return aggregated, header_metadata

def clean_data_for_ols_by_column(df, columns_to_check, config_path="config.json"):
    """
    Handles missing data based on predefined rules in the configuration file.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns_to_check (list): List of columns to check for missing values.
        config_path (str): Path to the main config file.

    Returns:
        pd.DataFrame: Updated DataFrame with missing values handled.
    """
    allowed_options = ['drop', 'mean', 'median', 'mode', 'zero', 'calc']

    # Load config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {"data": {"data_fitting": {}}}

    # Get missing data handling rules
    data_fitting_config = config.get("data", {}).get("data_fitting", {})

    for col in columns_to_check:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found in DataFrame, skipping.")
            continue

        # Get missing rows
        missing_urs = df.loc[df[col].isnull(), "UR"].tolist()
        if missing_urs:
            styled_print(f"⚠️ Missing values found in '{col}': {missing_urs}")

            # Get the pre-configured option
            option = data_fitting_config.get(col, None)

            if not option:
                styled_print(f"❓ Choose a method to handle missing data for '{col}': ['drop', 'mean', 'median', 'mode', 'zero', 'calc']")
                option = input("Enter choice: ").strip().lower()
                while option not in allowed_options:
                    option = input("Invalid choice. Please enter: 'drop', 'mean', 'median', 'mode', 'zero', 'calc': ").strip().lower()

                data_fitting_config[col] = option
                config["data"]["data_fitting"] = data_fitting_config
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

            # Apply the chosen option
            if isinstance(option, dict) or option == "calc":
                calc_conf = option if isinstance(option, dict) else None
                if calc_conf is None:
                    print(f"⚠️ Calculation configuration for '{col}' is missing, skipping calculation.")
                else:
                    first_input = calc_conf["first_input"]
                    second_input = calc_conf["second_input"]
                    op = calc_conf["operator"]
                    unit = calc_conf["unit"].lower()

                    if op != "-" or unit != "month":
                        print(f"⚠️ Only '-' and 'month' are supported for '{col}', skipping calculation.")
                    else:
                        df[first_input] = pd.to_datetime(df[first_input], errors="coerce")
                        df[second_input] = pd.to_datetime(df[second_input], errors="coerce")

                        df.loc[:, col] = df.apply(
                            lambda row: (row[first_input] - row[second_input]).days / 30 if pd.isnull(row[col]) and pd.notnull(row[first_input]) and pd.notnull(row[second_input]) else row[col],
                            axis=1
                        )
                        styled_print(f"✅ '{col}' values calculated using: {first_input} {op} {second_input} in {unit}s.")

            elif option == "drop":
                df = df.dropna(subset=[col])
                styled_print(f"✅ Rows with missing '{col}' have been dropped.")

            elif option == "mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df.loc[:, col] = df[col].fillna(df[col].mean())  # ✅ Best practice: No inplace=True
                    styled_print(f"✅ Missing '{col}' values filled with mean.")
                else:
                    print(f"⚠️ '{col}' is not numeric; skipping mean imputation.")

            elif option == "median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df.loc[:, col] = df[col].fillna(df[col].median())  # ✅ No inplace=True
                    styled_print(f"✅ Missing '{col}' values filled with median.")
                else:
                    print(f"U+26A0'{col}' is not numeric; skipping median imputation.")

            elif option == "mode":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df.loc[:, col] = df[col].fillna(mode_val.iloc[0])  # ✅ No inplace=True
                    styled_print(f"✅ Missing '{col}' values filled with mode.")
                else:
                    print(f"⚠️ '{col}' has no mode; skipping mode imputation.")

            elif option == "zero":
                df.loc[:, col] = df[col].fillna(0)  # ✅ No inplace=True
                styled_print(f"✅ Missing '{col}' values filled with zero.")

        else:
            print(f"✅ No missing data detected in '{col}'.")

    return df