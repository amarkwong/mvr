# How to Use This Tool

## Install
Following steps will executed in your `Terminal` app
- [ ] Install `brew` on your Mac if you haven't
  ```
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- [ ] Use `brew` to install python
  ```
    brew update
    brew install python
  ```
- [ ] Adding alias for python
  ```
    echo 'alias python="python3"' >> ~/.zshrc  # For zsh (default in macOS)
    echo 'alias pip="pip3"' >> ~/.zshrc
    source ~/.zshrc  # Apply changes
  ```
- [ ] update pip
  ```
  pip install --upgrade pip setuptools wheel
  ```

We should have python installed on our Mac now, next setup is to set up the virtual environment for python
- [ ] create the virtual environment and install the dependencies
```
python -m venv venv
source venv/bin/activate  # Activate the virtual environment
pip install -r requirements.txt
```

- [ ] finally, put your data.xlsx under the data folder, and we can run some stats
```
python main.py
```


## Data Requirements, Cleansing, Fitting, and Derivation

- **Data File Path:**  
  The path to the data file is specified in the `config.json` file under the "file" section. By default, the tool looks for a file named `data.xlsx` in the `data` folder.

- **Header Naming Convention:**  
  Each header should follow the pattern `column name #description`. The description can be omitted if the column name is self-explanatory. Otherwise, please include a clear description.

- **Data Formatting:**  
  The tool can handle some inconsistencies in data formatting. For example, if a column contains both percentage values and plain numbers in different rows, the functions `detect_percentage_format` and `standardize_numeric_columns` in `data.py` will help standardize these values.

- **Missing Data:**  
  Ideally, every data point should be present, but the tool implements a “smart” autofilling feature that lets the user choose how to handle missing data:
  - **drop:** Remove the row with the missing data.
  - **mean:** Replace the missing value with the average value.
  - **median:** Replace the missing value with the median value.
  - **mode:** Replace the missing value with the most common value.
  - **zero:** Replace the missing value with 0.
  - **calc:** Compute the missing value using specified rules (currently, only subtraction is supported for a given column).

- **Derived Data:**  
  New data columns can be created based on the raw data. Please refer to the "data" section in `config.json` for examples on how to set up derivation rules.

## Statistics

Currently, the tool supports the following analyses:

- Kaplan-Meier Survival Analysis
- Cox Regression
- OLS Regression
- Baseline Demographic Analysis

These settings can be toggled in the `stats` section of `config.json`. Note that some functionalities are still under testing.

## UI

The user interface is under development. Our goals for the UI are to:

- Provide configurable charts and tables.
- Output the charts and tables to a Markdown file.

Some features are partially working and still need further tuning.