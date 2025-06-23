# Appliance Energy Data Analytics

## Overview

This project provides tools for analyzing and visualizing appliance energy consumption data. The code includes functions for generating distribution graphs, correlation matrices, and scatter plots to help understand patterns in energy usage across different household appliances.

## Features

- Data loading and preprocessing from CSV files
- Visualization of per-column distributions (histograms for numerical data, bar charts for categorical data)
- Generation of correlation matrices to identify relationships between appliance energy consumption
- Creation of scatter matrix plots with density diagrams and correlation coefficients

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - mpl_toolkits.mplot3d

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/appliance-energy-analytics.git
cd appliance-energy-analytics
pip install -r requirements.txt
```

## Usage

The main functions provided in the codebase are:

1. `plotPerColumnDistribution(df, nGraphShown, nGraphPerRow)`: Creates histogram/bar charts for each column in the dataframe
2. `plotCorrelationMatrix(df, graphWidth)`: Generates a correlation matrix for the dataframe columns
3. `plotScatterMatrix(df, plotSize, textSize)`: Creates a scatter matrix with density plots and correlation coefficients

Example usage:

```python
# Load your data
df = pd.read_csv('path/to/your/data.csv')
df.dataframeName = 'your_data.csv'

# Plot distributions
plotPerColumnDistribution(df, 10, 5)  # Show 10 graphs, 5 per row

# Generate correlation matrix
plotCorrelationMatrix(df, 8)  # Use a graph width of 8

# Create scatter matrix
plotScatterMatrix(df, 15, 10)  # Plot size 15, text size 10
```

## Data Format

The code is designed to work with CSV files containing time-series data for various appliances. The example dataset includes the following columns:

- time: Timestamp of the measurement
- Various appliance columns (modem, cooker, rice_cooker, etc.): Power consumption in watts
- aggregate: Total power consumption

## Troubleshooting

- If you encounter the error `Number of rows must be a positive integer, not X`, ensure that the `nGraphRow` calculation in `plotPerColumnDistribution` results in an integer value.
- If you see `DataFrame.dropna() takes 1 positional argument but 2 were given`, update the `plotCorrelationMatrix` function to use keyword arguments: `df.dropna(axis='columns')` instead of `df.dropna('columns')`.

## Web app set up

A Flask web application to test your trained LSTM model for household power consumption prediction in real-time.
🚀 Quick Setup

1. Project Structure
   Create the following folder structure:
   lstm_web_app/
   ├── app.py # Flask backend
   ├── requirements.txt # Python dependencies
   ├── templates/
   │ └── index.html # Frontend HTML
   └── models/ # Your trained model files
   └── best_SlightlyImprovedLSTMModel_20250623_224633.pth
2. Installation
   bash# Create virtual environment
   python -m venv venv

# Activate virtual environment

# On Windows:

venv\Scripts\activate

# On macOS/Linux:

source venv/bin/activate

# Install dependencies

pip install -r requirements.txt 3. Add Your Model
Copy your trained model file to the models/ directory:

models/best_SlightlyImprovedLSTMModel_20250623_224633.pth

If you don't have the model file, the app will run in demo mode with a randomly initialized model. 4. Run the Application
bashpython app.py
Open your browser and navigate to: http://localhost:5000

## Contributing

Contributions to improve the codebase are welcome. Please feel free to submit a pull request or open an issue to discuss potential changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project utilizes the NILM (Non-Intrusive Load Monitoring) approach for analyzing appliance energy consumption.
- The example dataset contains measurements of various household appliances.
