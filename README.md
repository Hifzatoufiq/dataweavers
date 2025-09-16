
Stock Anomaly Detection - GameStop (GME)

This project detects anomalies in the stock price data of GameStop (GME) using multiple machine learning techniques. It provides insights into unusual stock movements and potential trading opportunities.

Features

Data retrieval using yfinance

Comprehensive Exploratory Data Analysis (EDA)

Multiple anomaly detection methods:

Z-Score

Isolation Forest

DBSCAN

LSTM Neural Networks

Autoencoder

Performance comparison of different methods

Interactive Flask web app for visualization

Humanized, user-friendly interface

Installation

Clone the repository:

git clone https://github.com/yourusername/stock-anomaly-detection.git
cd stock-anomaly-detection


Create a virtual environment (optional but recommended):

python -m venv venv
# Activate the environment:
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Install required packages:

pip install -r requirements.txt

Usage
Run Jupyter Notebook (for analysis)
jupyter notebook notebooks/Stock_Anomaly_Detection.ipynb

Launch Flask Web App (for interactive use)
python app.py


Then open your browser and visit:

http://127.0.0.1:5000

Project Structure
stock-anomaly-detection/
│
├─ app.py                 # Flask app
├─ templates/             # HTML templates
├─ static/                # CSS & JS files
├─ data/                  # Stock data storage
├─ models/                # Saved trained models
├─ notebooks/             # Jupyter notebook for analysis
├─ requirements.txt       # Required Python packages
└─ utils.py               # Data processing & anomaly detection functions

Methodology

Data Collection: Fetch stock data using yfinance.

Preprocessing: Clean data, handle missing values, calculate returns & volatility.

Exploratory Data Analysis (EDA): Visualize trends, volume, returns, and volatility.

Anomaly Detection Methods:

Z-Score

Isolation Forest

DBSCAN

LSTM

Autoencoder

Model Comparison: Evaluate performance using precision, recall, and F1-score.

Visualization: Display anomalies interactively in the Flask app.

Future Work

Add sentiment analysis or market indicators

Implement ensemble methods for better detection

Extend analysis to other stocks or real-time data


# Blog Post:
https://stockpulseaptech.blogspot.com/2025/09/stockpulse-detecting-stock-market.html


# video Link

https://www.mediafire.com/file/76bb7w226ggqg17/StockPulse.webm/file
