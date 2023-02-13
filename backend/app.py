from flask import Flask, jsonify, request
from catboost import CatBoostClassifier
import pandas as pd


app = Flask(__name__)

def predict_fraud(data):
    """"""
    model = CatBoostClassifier()
    model.load_model('model/model.h5')
    data = pd.DataFrame(data)
    pred = model.predict_proba(data)
    return pred[0][1]

@app.route("/predict-fraud", methods=['GET'])
def predict():
    labels = ['Time', 'Amount', 'use_chip', 'MCC', 'transaction_error', 'age',
              'ret_age', 'Gender', 'Zipcode', 'yearly_income', 'total_debt', 'fico_score']
    data = {label:[request.args.get(label, 0)] for label in labels}
    prediction = predict_fraud(data)
    return jsonify({'pred':prediction})


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=5000, debug=True)
