import pandas as pd
import json

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
            print('col',col_str)
            actual, desc = col_str.split('#', 1)
            actual = actual.strip()
            desc = desc.strip()
            header_names.append(actual)
            descriptions[actual] = desc
        else:
            header_names.append(col_str)
    return header_names, descriptions

def to_list(x):
    return list(x)

def first_non_null(x):
    nonnull = x.dropna()
    return nonnull.iloc[0] if not nonnull.empty else None

def combine_gene_info(row):
    gene_records = []
    for gene, vaf, tier, var_desc in zip(
        row["Gene"],
        row["VAF% G1"],
        row["Tier"],
        row["Variant description"]
    ):
        gene_records.append({
            "name": gene,
            "VAF% G1": vaf,
            "Tier": tier,
            "Variant description": var_desc
        })
    return gene_records

def main():
    # Read the Excel file and forward-fill vertical merges.
    df = pd.read_excel("data/data1.xlsx", engine="openpyxl")
    df = df.ffill(axis=0)

    # Debug: print raw header names.
    raw_headers = list(df.columns)
    
    # Process headers: if a header contains a '#', split it.
    new_columns, header_metadata = process_headers(raw_headers)
    
    # Replace the DataFrame columns with the processed header names.
    df.columns = new_columns

    # Define gene-related columns.
    gene_columns = ["Gene", "VAF% G1", "Tier", "Variant description"]
    
    # Build an aggregation dictionary for grouping.
    agg_dict = {}
    for col in df.columns:
        if col == "UR":
            continue  # Grouping key.
        if col in gene_columns:
            agg_dict[col] = to_list
        else:
            agg_dict[col] = first_non_null
    
    # Group by "UR" and aggregate.
    aggregated = df.groupby("UR", as_index=False).agg(agg_dict)
    
    # Combine gene-related columns into a single nested list.
    aggregated["Gene"] = aggregated.apply(combine_gene_info, axis=1)
    aggregated = aggregated.drop(columns=["VAF% G1", "Tier", "Variant description"])
    print(aggregated)
    
    # Convert the aggregated DataFrame to JSON.
    json_output = aggregated.to_json(orient="records", indent=2)
    
    with open("merged_output.json", "w") as f:
        f.write(json_output)

if __name__ == "__main__":
    main()