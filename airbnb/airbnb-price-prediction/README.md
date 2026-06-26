# NYC Airbnb Price Prediction and Market Analysis

This project aims to build a predictive model to forecast Airbnb prices in New York City using the open dataset for 2019. The project performs exploratory data analysis (EDA), data cleaning, feature engineering, and regression modeling. It highlights the major factors driving listing prices in NYC and provides host-focused pricing strategies.

## Tech Stack
*   **Python 3.11**
*   **Pandas & NumPy** for data manipulation
*   **Scikit-Learn** for preprocessing and machine learning modeling
*   **Matplotlib & Seaborn** for custom data visualization
*   **Sweetviz & Autoviz** for automated data profiling
*   **Jupyter Notebook** for interactive exploration and training

## Folder Structure
```text
airbnb-price-prediction/
│
├── data/
│   ├── raw/                       # Raw downloaded dataset
│   └── cleaned/                   # Cleaned and feature engineered CSV data
│
├── src/
│   ├── data_cleaning.py           # Data download, cleaning, and feature engineering
│   ├── visualization.py           # Custom visualization plotting functions
│   └── modeling.py                # Preprocessing pipelines, training, and evaluation
│
├── notebooks/
│   ├── exploration.ipynb          # Automated EDA and custom visualization notebook
│   └── modeling.ipynb             # ML training, model evaluation, and diagnostics
│
├── reports/
│   ├── figures/                   # Saved custom matplotlib/seaborn plots
│   ├── autoviz/                   # Automatically generated Autoviz plots
│   └── sweetviz_report.html       # Sweetviz HTML interactive data profile
│
├── README.md                      # Project documentation
└── requirements.txt               # Required packages list
```

## How to Run

1.  **Clone/Open the Repository**:
    Navigate to the workspace directory.

2.  **Install Required Libraries**:
    Ensure you have python installed. Run the command:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Analysis & Modeling**:
    You can run the notebooks in the `notebooks/` directory in order:
    1.  `notebooks/exploration.ipynb`
    2.  `notebooks/modeling.ipynb`
    
    Alternatively, you can run them programmatically in the shell:
    ```bash
    jupyter nbconvert --to notebook --execute --inplace notebooks/exploration.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/modeling.ipynb
    ```

## Key Findings & Pricing Recommendations

1.  **Borough Premia**: Geographical location is the single largest price driver. Manhattan listings command a substantial premium (median price ~$150), followed by Brooklyn (~$90), while Queens, the Bronx, and Staten Island represent value-oriented options (median price ~$60-$70).
2.  **Room Type Impact**: Entire home/apt pricing is nearly double that of private rooms. Shared rooms are priced the lowest. Hosts should consider converting shared configurations into private or entire spaces where possible.
3.  **Model Performance**: Our best model, the **Gradient Boosting Regressor**, predicts NYC Airbnb prices with a Mean Absolute Error (MAE) of **~$45** on the hold-out test set, capturing non-linear interactions (such as the combined impact of room type and location coordinates) much better than simple linear models.
