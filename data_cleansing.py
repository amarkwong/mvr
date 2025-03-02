import pandas as pd
import os
import json
import pandas as pd

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
    
    # Debug: print the lengths and contents of the lists for this row.
    # print(f"Debug combine_gene_info for UR {row['UR']}:")
    # print(f"  Genes: {genes} (len={len(genes)})")
    # print(f"  VAF% G1: {vafs} (len={len(vafs)})")
    # print(f"  Tier: {tiers} (len={len(tiers)})")
    # print(f"  Variant description: {var_descs} (len={len(var_descs)})")
    
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
    Load Excel data and perform data cleansing steps:
      - Use specified NA strings.
      - Process headers.
      - Sort by patient identifier ('UR') so rows remain in order.
      - Do NOT apply global forward-fill to gene-related columns so that each raw row is preserved.
      - Build an aggregation dictionary:
            * For gene-related columns, collect all values (using tolist()).
            * For non-gene columns, take the first value.
      - Group by 'UR' and aggregate.
      - Print debug information on group sizes and for a selected UR (e.g., PA177925).
      - Combine gene-related columns into a nested "Gene" list.
      - Write the aggregated DataFrame to a JSON file.
      
    Returns:
      - Aggregated DataFrame.
      - Header metadata dictionary.
    """
    na_values = ["NA", "na", "N/A", "n/a", "N/a"]
    df = pd.read_excel(file_path, engine="openpyxl", na_values=na_values)
    
    # Process headers.
    raw_headers = list(df.columns)
    new_columns, header_metadata = process_headers(raw_headers)
    # Sort by patient identifier.
    if "UR" in df.columns:
        df = df.sort_values("UR")
    
    # Debug: print group sizes (number of rows per UR) in the raw data.
    if "UR" in df.columns:
        group_sizes = df.groupby("UR").size()
        print("Debug: Group sizes in raw data:")
        print(group_sizes)
    
    # Define gene-related columns.
    gene_columns = ["Gene", "VAF% G1", "Tier", "Variant description"]
    # Note: Do NOT forward-fill these columns here.
    
    # Build aggregation dictionary.
    agg_dict = {}
    for col in df.columns:
        if col == "UR":
            continue  # Grouping key.
        if col in gene_columns:
            # Use a lambda that captures the column and returns all values as a list.
            agg_dict[col] = lambda x, col=col: x.tolist()
        else:
            agg_dict[col] = lambda x: x.iloc[0]
    
    aggregated = df.groupby("UR", as_index=False).agg(agg_dict)
    
    # Debug: For a selected UR (e.g., PA177925), print the aggregated gene-related lists.
    debug_ur = "PA177925"
    if debug_ur in aggregated["UR"].values:
        agg_subset = aggregated[aggregated["UR"] == debug_ur]
        print(f"Debug: Aggregated gene columns for UR {debug_ur}:")
        for col in gene_columns:
            print(f"  {col}: {agg_subset.iloc[0][col]} (len={len(agg_subset.iloc[0][col])})")
    else:
        print(f"Debug: UR {debug_ur} not found in aggregated data.")
    
    # Combine gene-related columns into a single nested "Gene" list.
    aggregated["Gene"] = aggregated.apply(combine_gene_info, axis=1)
    aggregated = aggregated.drop(columns=["VAF% G1", "Tier", "Variant description"])
    
    # Write aggregated DataFrame to JSON for inspection.
    aggregated.to_json("merged_output.json", orient="records", indent=2)
    
    return aggregated, header_metadata