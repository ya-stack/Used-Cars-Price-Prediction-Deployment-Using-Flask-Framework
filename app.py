import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
enc = pickle.load(open('encoder.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    encoded_features = list(enc.transform(np.array(int_features[-4:]).reshape(1, -1))[0])
    to_predict = np.array(int_features[:-4] + encoded_features)
    to_predict = to_predict.astype('float64')
    test_result = model.predict(to_predict.reshape(1, -1))
    output = test_result[0]
    print(output)
    return render_template('index.html', prediction_text='Price of used cars will be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)