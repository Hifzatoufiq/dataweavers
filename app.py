from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

app = Flask(__name__)

def load_data(file_path, start="2000-01-01", end="2023-12-31"):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    data = data.loc[start:end].copy()

    if data.empty or len(data) < 20:
        return pd.DataFrame()

    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Upper_BB'] = data['MA20'] + (rolling_std * 2)
    data['Lower_BB'] = data['MA20'] - (rolling_std * 2)
    return data.dropna()

def detect_zscore(data, threshold=3):
    z_scores = np.abs((data['Close'] - data['Close'].mean()) / data['Close'].std())
    return z_scores > threshold

def detect_iforest(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['Close','Volume','Returns','Volatility']])
    model = IsolationForest(contamination=0.01, random_state=42)
    return model.fit_predict(X) == -1

def detect_dbscan(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['Close','Volume','Returns','Volatility']])
    model = DBSCAN(eps=0.5, min_samples=5)
    return model.fit_predict(X) == -1

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/anomalies", methods=["POST"])
def anomalies():
    start_date = request.json.get("start_date", "2000-01-01")
    end_date = request.json.get("end_date", "2023-12-31")
    data = load_data("clean_gme_data.csv", start=start_date, end=end_date)

    if data.empty:
        return jsonify({"error": "Not enough data"}), 400

    zscore_anomalies = detect_zscore(data)
    iforest_anomalies = detect_iforest(data)
    dbscan_anomalies = detect_dbscan(data)

    response = {
        "dates": data.index.strftime("%Y-%m-%d").tolist(),
        "close": data['Close'].tolist(),
        "upper_bb": data['Upper_BB'].tolist(),
        "lower_bb": data['Lower_BB'].tolist(),
        "zscore_anomalies": [int(x) for x in zscore_anomalies],
        "iforest_anomalies": [int(x) for x in iforest_anomalies],
        "dbscan_anomalies": [int(x) for x in dbscan_anomalies]
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
