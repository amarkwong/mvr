import pandas as pd
import os
import json

def process_headers(columns):
    header_names = []
    descriptions = {}
    for col in columns:
        col_str = str(col).strip()
        if '#' in col_str:
            actual, desc = col_str.split('#', 1)
            actual = actual.strip().lstrip('*')
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