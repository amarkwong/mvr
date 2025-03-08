from data import data_cleansing, generate_metadata_mapping, data_derive, data_fitting
from stats import baseline_demographic, multivariate_linear_regression, cox_regression, km_estimate
from ui import dual_axis_histogram_box_chart, fetch_boxplot_data, styled_print, display_demographic_data, plot_km_survival_curves
from lifelines import CoxPHFitter
import pandas as pd
import json

def main():
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    # Load and clean dataset
    file_path = "data/data.xlsx"
    aggregated, header_metadata = data_cleansing(file_path)

    data_fitting(aggregated,['Dx OS'])
    data_fitting(aggregated,['Ferritin','TF Sats','BM Iron stores'])

    # derive data for Cox and KM
    aggregated, header_metadata = data_derive(aggregated)

    print(aggregated[["TF Sats", "Ferritin", "Serum Iron Class"]].head())

    print(aggregated[aggregated['TF Sats']<1])

    # Count events (deaths) and censoring
    group = "BM Iron stores Class"  # Change if needed
    group_name = "reduced"  # Name of the class in the dataset

    subset = aggregated[aggregated[group] == 0]
    print("subset",subset)
    deaths = subset[subset["Death"] == 1].shape[0]
    censored = subset[subset["Death"] == 0].shape[0]
    total = subset.shape[0]

    print(f"Group: {group_name}")
    print(f"Total Patients: {total}")
    print(f"Deaths: {deaths}")
    print(f"Censored: {censored}")

    # Generate metadata lookup
    metadata_lookup = generate_metadata_mapping(header_metadata)

    # Cox regression
    # cox_model = cox_regression(aggregated)
    # cox_model.print_summary()


    # Kaplan-Meier
    # ? Compute KM estimates (logic-only, no UI)
    km_results = km_estimate(aggregated)

    # ? Plot survival curves separately (UI-only)
    plot_km_survival_curves(km_results)

    # Extract UI settings for colors
    histogram_colors = config["ui"].get("histogram", {}).get("color", {})
    boxplot_settings = config["ui"].get("boxplot", {})

    # Compute baseline demographics for 'table' mode
    demo_stats = baseline_demographic(aggregated, config)

    # Display demographic data dynamically based on `config.json`
    for demographic in config["stats"].get("baseline_demographic", []):
        if not demographic.get("enabled", True):
            continue  # Skip if disabled

        display_mode = demographic.get("display_mode", "table")
        histogram_variable = demographic.get("histogram_variable")
        numeric_variable = demographic.get("numeric_variable")

        if display_mode == "table":
            display_demographic_data(config, aggregated, histogram_variable, numeric_variable, demo_stats)

        elif display_mode == "chart":
            if histogram_variable in aggregated.columns and numeric_variable in aggregated.columns:
                print(f"\n📊 Displaying {histogram_variable} vs {numeric_variable} as a Chart:")

                # ✅ Generate histogram data
                histogram_data = aggregated[histogram_variable].value_counts().to_dict()

                # ✅ Check if metadata exists, else use raw values
                metadata_mapping = metadata_lookup.get(histogram_variable, {})
                histogram_data = {metadata_mapping.get(k, str(k)): v for k, v in histogram_data.items()}

                # ✅ Apply metadata mapping to DataFrame for Seaborn boxplot
                df_modified = aggregated.copy()
                if metadata_mapping:
                    df_modified[histogram_variable] = df_modified[histogram_variable].map(metadata_mapping)

                # ✅ Ensure color mapping falls back to "default"
                bar_colors = {
                    metadata_mapping.get(k, str(k)): histogram_colors.get(metadata_mapping.get(k, str(k)), histogram_colors.get("default", "#AAAAAA"))
                    for k in histogram_data.keys()
                }

                # Render chart with updated labels and color settings
                dual_axis_histogram_box_chart(
                    histogram_data,
                    df_modified,  # Pass modified df with human-readable labels
                    x_label=histogram_variable,
                    y1_label="Patient Count",
                    y2_label=numeric_variable,
                    title=f"{histogram_variable} vs. {numeric_variable} Distribution",
                    color_config=bar_colors,  # ✅ Apply color settings from config
                    box_opacity=boxplot_settings.get("opacity", 0.5),  # ✅ Apply boxplot opacity
                    box_width=boxplot_settings.get("width", 0.5)  # ✅ Apply boxplot width
                )
            else:
                print(f"⚠️ Warning: One or more columns missing for {histogram_variable} vs {numeric_variable}. Skipping.")

    # Run Multivariate Linear Regression
    styled_print("Multivariate Linear Regression:", color="blue", style="bold")
    ols_settings = config["stats"].get("ols_setting", {})
    x_columns = ols_settings.get("x_columns", [])
    y_column = ols_settings.get("y_column", "")

    if x_columns and y_column in aggregated.columns:
        multivariate_linear_regression(aggregated, x_columns, y_column)
    else:
        print("⚠️ Warning: OLS regression skipped due to missing columns.")

if __name__ == "__main__":
    main()