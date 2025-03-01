import pandas as pd

def process_headers(columns):
    """
    Process column headers so that if a header contains a '#' character,
    the part before '#' is used as the actual header and the part after '#' is stored as a description.
    Returns:
      - A list of new column names.
      - A dictionary mapping each new column name to its description.
    """
    header_names = []
    descriptions = {}
    for col in columns:
        # Force conversion to string and strip extra spaces.
        col_str = str(col).strip()
        if '#' in col_str:
            # Split on the first '#' character.
            actual, desc = col_str.split('#', 1)
            actual = actual.strip().lstrip('*')
            desc = desc.strip()
            header_names.append(actual)
            descriptions[actual] = desc
        else:
            header_names.append(col_str.lstrip('*'))
    return header_names, descriptions

def to_list(x):
    return list(x)

def first_non_null_in_group(x):
    """
    Return the first non-null value in the Series x.
    If every value is null, return None.
    """
    for val in x:
        if pd.notnull(val):
            return val
    return None

def combine_gene_info(row):
    """
    Combine gene-related columns into a nested list of dictionaries.
    Skip any gene record where the gene value is missing or equals 0 (or "0").
    """
    gene_records = []
    for gene, vaf, tier, var_desc in zip(
        row["Gene"],
        row["VAF% G1"],
        row["Tier"],
        row["Variant description"]
    ):
        # Skip if gene is missing, NaN, or equals 0 (as numeric or string)
        if pd.isna(gene) or gene == 0 or str(gene).strip() == "0":
            continue
        gene_records.append({
            "name": gene,
            "VAF% G1": vaf,
            "Tier": tier,
            "Variant description": var_desc
        })
    return gene_records

def load_and_clean_data(file_path):
    """
    Load Excel data, perform data cleansing including:
      - Treat defined strings as missing.
      - Forward-fill vertical merges.
      - Process headers with inline comments.
      - Aggregate data by UR and combine gene-related columns into a nested list.
      
    Returns:
      - Aggregated DataFrame.
      - Header metadata dictionary.
    """
    # Define missing values.
    na_values = ["NA", "na", "N/A", "n/a", "N/a"]
    df = pd.read_excel(file_path, engine="openpyxl", na_values=na_values)
    df = df.ffill(axis=0)
    
    # Process headers.
    raw_headers = list(df.columns)
    new_columns, header_metadata = process_headers(raw_headers)
    df.columns = new_columns
    
    # Define gene-related columns.
    gene_columns = ["Gene", "VAF% G1", "Tier", "Variant description"]
    
    # Build aggregation dictionary for grouping.
    agg_dict = {}
    for col in df.columns:
        if col == "UR":
            continue  # Grouping key.
        if col in gene_columns:
            agg_dict[col] = to_list
        else:
            agg_dict[col] = first_non_null_in_group
    
    aggregated = df.groupby("UR", as_index=False).agg(agg_dict)
    aggregated["Gene"] = aggregated.apply(combine_gene_info, axis=1)
    aggregated = aggregated.drop(columns=["VAF% G1", "Tier", "Variant description"])
    
    return aggregated, header_metadata