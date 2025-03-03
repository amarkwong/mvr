from data_cleansing import load_and_clean_data
from statistics import baseline_demographic, multivariate_linear_regression
from ui import plot_histogram, print_table_markdown, styled_print, dual_axis_histogram_line_chart
import pandas as pd

def main():
    # Update with your actual Excel file path.
    file_path = "data/data.xlsx"
    aggregated, header_metadata = load_and_clean_data(file_path)

    # Define baseline variables and their corresponding metrics.
    baseline_vars = ['Gender', 'Age at dx', 'Gene']
    baseline_metric = ['histogram', 'median', 'count histogram']
    
    # Compute baseline demographic statistics.
    demo_stats = baseline_demographic(aggregated, baseline_vars, baseline_metric)

    print('demo_stats',demo_stats)
    
    styled_print("Baseline Demographic Stats:", color="blue", style="bold")
    for var, stat in demo_stats.items():
        print(f"\n{var}:")
        print(stat)
    
    # Example: Plot histogram for Gender (if available).
    if 'Gender' in demo_stats and isinstance(demo_stats['Gender'], dict):
        dual_axis_histogram_line_chart(aggregated,'Gender','Age at dx')
    
    # Example: For the Gene count histogram, convert to a DataFrame and print as a Markdown table.
    if 'Gene' in demo_stats and isinstance(demo_stats['Gene'], dict):
        gene_hist_df = pd.DataFrame(list(demo_stats['Gene'].items()), columns=['Gene Defects Count', 'Patient Count'])
        # age_avg_df = .groupby(list(demo_stats['Gene'].items()))['Age at dx'].mean().reset_index()
        # gene_plot_df = gene_hist_df.merge(age_avg_df, on="Gene Defects Count", how="left")
        # styled_print("\nGene Defect Histogram Table (Markdown):",color="yellow")
        dual_axis_histogram_line_chart(gene_plot_df, "Gene Defects Count", "Age at dx")
        # print_table_markdown(gene_hist_df)

    styled_print("Multivariate Linear Regression:", color="blue", style="bold")
    x_columns = ['Age at dx','ELN 2022 Risk','BM Iron stores','Ferritin','TF Sats','Allograft']
    y_column = 'Dx OS'
    multivariate_linear_regression(aggregated, x_columns, y_column)

if __name__ == "__main__":
    main()