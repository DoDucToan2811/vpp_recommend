from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.ensemble import IsolationForest

app = Flask(__name__)
CORS(app)

DB_CONFIG = {
    "server": "localhost",
    "database": "OSManagement",
    "driver": "ODBC Driver 17 for SQL Server"
}

def get_engine():
    conn_str = (
        f"mssql+pyodbc://@{DB_CONFIG['server']}/{DB_CONFIG['database']}?"
        f"driver={DB_CONFIG['driver'].replace(' ', '+')}&trusted_connection=yes"
    )
    return create_engine(conn_str)

def load_data():
    engine = get_engine()

    product_requests = pd.read_sql("SELECT ProductID, RequestID, Quantity FROM Product_Requests WHERE IsDeleted = 0", engine)
    products = pd.read_sql("SELECT ProductID, Name FROM Products WHERE IsDeleted = 0", engine)
    users = pd.read_sql("SELECT UserID, Department FROM Users WHERE IsDeleted = 0", engine)
    
    # 👉 Only include approved requests
    requests = pd.read_sql("""
        SELECT RequestID, UserID 
        FROM Requests 
        WHERE IsDeleted = 0 AND IsApprovedBySupLead = 1
    """, engine)

    merged = product_requests.merge(requests, on="RequestID")
    merged = merged.merge(users, on="UserID")
    
    return merged, products


def train_models(data):
    # Prepare encoders
    le_dep = LabelEncoder()
    le_prod = LabelEncoder()
    
    data['Department_enc'] = le_dep.fit_transform(data['Department'])
    data['ProductID_enc'] = le_prod.fit_transform(data['ProductID'])

    # Classification model
    X_cls = data[['Department_enc']]
    y_cls = data['ProductID_enc']
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_cls, y_cls)

    # Regression model
    X_reg = data[['Department_enc', 'ProductID_enc']]
    y_reg = data['Quantity']
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_reg, y_reg)

    return clf, reg, le_dep, le_prod
# ✅ Global storage for models & detection results
detection_models = {
    "price_model": None,
    "quantity_model": None,
    "price_abnormal_ids": set(),
    "quantity_abnormal_ids": set()
}

def train_abnormality_models():
    engine = get_engine()

    requests_df = pd.read_sql("""
        SELECT r.RequestID, r.TotalPrice, r.UserID, u.Department
        FROM Requests r
        JOIN Users u ON r.UserID = u.UserID
        WHERE r.IsDeleted = 0 AND r.IsApprovedByDepLead = 1 AND r.IsProcessedByDepLead = 1
    """, engine)

    product_requests_df = pd.read_sql("""
        SELECT RequestID, ProductID, Quantity
        FROM Product_Requests
        WHERE IsDeleted = 0
    """, engine)

    products_df = pd.read_sql("""
        SELECT ProductID, Name
        FROM Products
        WHERE IsDeleted = 0
    """, engine)

    # ---- Train price anomaly model
    price_model = IsolationForest(contamination=0.1, random_state=42)
    requests_df['PriceAnomaly'] = price_model.fit_predict(requests_df[['TotalPrice']])
    abnormal_price_ids = set(requests_df[requests_df['PriceAnomaly'] == -1]['RequestID'])

    # ---- Train quantity anomaly model
    merged = requests_df.merge(product_requests_df, on='RequestID')
    merged = merged.merge(products_df, on='ProductID')

    quantity_model = IsolationForest(contamination=0.1, random_state=42)
    merged['QuantityAnomaly'] = quantity_model.fit_predict(merged[['Quantity']])
    abnormal_quantity_ids = set(merged[merged['QuantityAnomaly'] == -1]['RequestID'])

    # ✅ Save to global variable
    detection_models['price_model'] = price_model
    detection_models['quantity_model'] = quantity_model
    detection_models['price_abnormal_ids'] = abnormal_price_ids
    detection_models['quantity_abnormal_ids'] = abnormal_quantity_ids
@app.route('/retrain_anomaly_models', methods=['POST'])
def retrain_anomaly_models():
    train_abnormality_models()
    return '', 200

@app.route('/recommend_ml', methods=['GET'])
def recommend_ml():
    department = request.args.get('department')
    if not department:
        return jsonify({"error": "Missing 'department' parameter"}), 400

    data, product_info = load_data()
    clf, reg, le_dep, le_prod = train_models(data)

    if department not in le_dep.classes_:
        return jsonify([])

    # Predict product probabilities
    dep_id = le_dep.transform([department])[0]
    proba = clf.predict_proba([[dep_id]])[0]
    top_n = 3
    top_indices = np.argsort(proba)[::-1][:top_n]
    top_product_ids = le_prod.inverse_transform(top_indices)

    results = []
    for pid_enc, prob in zip(top_indices, proba[top_indices]):
        pid = le_prod.inverse_transform([pid_enc])[0]
        product_name = product_info[product_info['ProductID'] == pid]['Name'].values[0]

        # Predict quantity using regressor
        predicted_qty = int(round(reg.predict([[dep_id, pid_enc]])[0]))
        predicted_qty = max(predicted_qty, 1)  # Avoid 0

        results.append({
            "Product": product_name,
            "Quantity": predicted_qty,
            "Chance": round(prob, 3)
        })

    return jsonify(results)


from scipy.stats import zscore

@app.route('/detect_abnormal_requests', methods=['GET'])
def detect_abnormal_requests():
    engine = get_engine()

    # Load tables
    requests_df = pd.read_sql("""
        SELECT r.RequestID, r.TotalPrice, r.UserID, u.Department
        FROM Requests r
        JOIN Users u ON r.UserID = u.UserID
        WHERE r.IsDeleted = 0 AND r.IsApprovedByDepLead = 1 AND r.IsProcessedByDepLead = 1
    """, engine)

    product_requests_df = pd.read_sql("""
        SELECT RequestID, ProductID, Quantity
        FROM Product_Requests
        WHERE IsDeleted = 0
    """, engine)

    products_df = pd.read_sql("""
        SELECT ProductID, Name
        FROM Products
        WHERE IsDeleted = 0
    """, engine)

    # --- Abnormal TotalPrice Detection (AI)
    price_data = requests_df[['TotalPrice']]
    price_model = IsolationForest(contamination=0.1, random_state=42)
    requests_df['PriceAnomaly'] = price_model.fit_predict(price_data)

    abnormal_price = requests_df[requests_df['PriceAnomaly'] == -1]

    # --- Abnormal Quantity Detection (AI)
    merged = requests_df.merge(product_requests_df, on='RequestID')
    merged = merged.merge(products_df, on='ProductID')

    quantity_data = merged[['Quantity']]
    quantity_model = IsolationForest(contamination=0.1, random_state=42)
    merged['QuantityAnomaly'] = quantity_model.fit_predict(quantity_data)

    abnormal_quantity = merged[merged['QuantityAnomaly'] == -1]

    # Return results
    abnormal_combined = {
        "abnormal_total_price": abnormal_price[['RequestID', 'TotalPrice', 'Department']].to_dict(orient='records'),
        "abnormal_quantity": abnormal_quantity[['RequestID', 'Name', 'Quantity', 'Department']].rename(columns={'Name': 'Product'}).to_dict(orient='records')
    }

    return jsonify(abnormal_combined)
@app.route('/check_request_abnormality', methods=['GET'])
def check_request_abnormality():
    request_id = request.args.get('request_id')
    if not request_id:
        return jsonify({"error": "Missing 'request_id' parameter"}), 400

    try:
        request_id = int(request_id)
    except:
        return jsonify({"error": "Invalid request_id"}), 400

    result = {
        "RequestID": request_id,
        "IsAbnormal": False,
        "AbnormalTypes": []
    }

    if request_id in detection_models['price_abnormal_ids']:
        result["IsAbnormal"] = True
        result["AbnormalTypes"].append("TotalPrice")

    if request_id in detection_models['quantity_abnormal_ids']:
        result["IsAbnormal"] = True
        result["AbnormalTypes"].append("Quantity")

    return jsonify(result)


if __name__ == '__main__':
    train_abnormality_models()
    app.run(debug=True)