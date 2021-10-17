import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scale=pickle.load(open('scaling_model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    print(final_features)
    final_features[0]=(final_features[0]*800)/360
    print(final_features)
    prediction = model.predict(scale.transform([final_features]))
    print(prediction)
    if int(prediction[0])==1:
        output="SELECTED"
    else:
        output="NOT SELECTED"
    print(output)
    return render_template('index1.html',prediction_text='The status of the admission: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)


