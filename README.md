# DUT Synthetic Population Generation

This project generates synthetic population data for **Daejeon (대전)** and **Sejong (세종)**, South Korea, based on travel survey data. It utilizes **CTAB-GAN-Plus**, a conditional GAN-based tabular data generator, to create realistic synthetic datasets while preserving the statistical properties of the original sensitive data.

## Features

- **Synthetic Data Generation**: Uses CTAB-GAN-Plus to generate high-quality synthetic tabular data.
- **Hyperparameter Tuning**: Integrated **Optuna** for automated hyperparameter optimization to find the best model configuration.
- **Data Quality Analysis**:
    - **Statistical Comparison**: compares mean, standard deviation, and distributions of continuous variables (e.g., Age) between original and synthetic data.
    - **Categorical Distribution**: Visualizes and calculates differences in categorical variable distributions (e.g., Province, Administrative District).
    - **Correlation Matrix**: Compares feature correlations to ensure relationships are preserved.
    - **Similarity Scores**: Calculates Wasserstein distance and other metrics to quantify data similarity.
- **Visualization**: Automatically generates plots for distributions, correlations, and quality reports.

## Prerequisites

The project requires Python and the following libraries:

- `pandas`
- `numpy`
- `torch` (PyTorch)
- `plotly`
- `kaleido` (for Plotly image export)
- `optuna`
- `scikit-learn`
- `sdv` (Synthetic Data Vault)
- `matplotlib`
- `seaborn`
- `scipy`
- `dython`

You can install the dependencies using pip:

```bash
pip install pandas numpy torch plotly kaleido optuna scikit-learn sdv matplotlib seaborn scipy dython
```

*Note: You may need to install specific versions compatible with your CUDA environment if you plan to use GPU acceleration.*

## Project Structure

- `generate.py`: The main script that:
    1. Loads the preprocessed travel survey data.
    2. Filters data for Daejeon and Sejong.
    3. Runs Optuna to tune CTAB-GAN-Plus hyperparameters.
    4. Trains the best model and generates synthetic samples.
    5. Performs comprehensive evaluation and visualization.
- `CTAB-GAN-Plus/`: Directory containing the CTAB-GAN-Plus model implementation.
    - `model/ctabgan.py`: Main model class.
    - `model/synthesizer/`: GAN synthesizer implementation.
    - `model/pipeline/`: Data preprocessing pipeline.

## Usage

1. **Prepare Data**:
    - Ensure the preprocessed travel survey CSV file is in the correct location.
    - *Note: The current script uses hardcoded paths like `C:\Users\ADMIN\...`. You will likely need to modify `original_data_path` and `ctab_gan_plus_path` in `generate.py` to match your local environment.*

2. **Run Generation**:
    ```bash
    python generate.py
    ```

3. **Output**:
    - **Synthetic Data**: Saved as `travel_survey_generated.csv`.
    - **Visualizations**: Saved in `outputs/optimal_ctabgan+/visualizations/`.
    - **Reports**: A text report `quality_analysis_report.txt` summarizing the data quality scores.

## Configuration

The script currently includes hardcoded filters for specific regions:
- `home_province == '대전광역시'` (Daejeon)
- `home_province == '세종특별자치시'` (Sejong)

If you wish to use this with different datasets, ensure you update the filtering logic in `generate.py`.

## Acknowledgements

This project builds upon the **CTAB-GAN-Plus** architecture for tabular data synthesis.
