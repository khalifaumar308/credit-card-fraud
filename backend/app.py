from flask import Flask, jsonify, request
from flask_cors import CORS
from catboost import CatBoostClassifier
import pandas as pd


app = Flask(__name__)
CORS(app)


def cvt_to_numb(label, value):
    print(label, value)
    conv_dict =  {
        'transaction_error':{'None':0, 'Insufficient Balance':1, 'Bad PIN':2, 
                    'Bad CVV':3, 'Bad Expiration':4, 'Bad Card Number':5, 'Technical Glitch':6},
        'use_chip':{'Swipe Transaction':0,'Chip Transaction':1, 'Online Transaction':3},
        'Gender':{'Male':0,'Female':1}}
    return conv_dict[label][value]

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
    data = {label:([request.args.get(label, 0)] if label not in ('Gender','transaction_error','use_chip') 
                   else [cvt_to_numb(label,request.args.get(label,0))])
             for label in labels}
    return jsonify(f'{predict_fraud(data)}')


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=5000, debug=True)
