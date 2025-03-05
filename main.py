import json
import pandas as pd
from data import load_and_clean_data, generate_metadata_mapping
from stats import baseline_demographic, multivariate_linear_regression
from ui import display_demographic_data

def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    # Load and clean dataset
    file_path = "data/data.xlsx"
    aggregated, header_metadata = load_and_clean_data(file_path)

    # Ensure 'Gene Count' column exists
    if "Gene" in aggregated.columns:
        aggregated["Gene Count"] = aggregated["Gene"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Generate metadata lookup
    metadata_lookup = generate_metadata_mapping(header_metadata)

    # Compute baseline demographics
    demo_stats = baseline_demographic(aggregated, config)

    # ✅ Generate Markdown report
    with open("analysis.md", "w") as md:
        md.write("# 📊 Analysis Report\n\n## 1️⃣ Baseline Demographic\n")

    for demographic in config["stats"].get("baseline_demographic", []):
        if not demographic.get("enabled", True):
            continue

        display_demographic_data(
            config, 
            aggregated, 
            demographic["histogram_variable"], 
            demographic["numeric_variable"], 
            demo_stats, 
            save_file="analysis.md", 
            metadata_lookup=metadata_lookup
        )

    print("✅ Analysis report saved as `analysis.md`")

if __name__ == "__main__":
    main()