import pandas as pd

def main():
    # Read the Excel file; assumes row 1 is the header.
    df = pd.read_excel("data/data.xlsx", engine="openpyxl")
    
    # Forward-fill missing values in each column (fills NaN caused by vertical merges).
    df = df.ffill(axis=0)
    
    # Display the DataFrame to verify the vertical merge values are filled correctly.
    print("DataFrame:")
    print(df)
    
    # Convert the DataFrame to JSON (records orient) with pretty indentation.
    json_output = df.to_json(orient="records", indent=2)
    print("\nJSON output:")
    print(json_output)
    
    # Optionally, save the JSON output to a file.
    with open("output.json", "w") as f:
        f.write(json_output)

if __name__ == "__main__":
    main()