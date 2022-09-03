import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle

## create Flask app

app  = Flask(__name__)

# load the pickle model

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features  = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html",prediction_text = "The Electrical {}".format(prediction))

#@app.route("/predict", methods = ["POST"] )
#def predict():
 #   json_ = request.json
   # query_df = pd.DataFrame(json_)
   # prediction = model.predict(query_df)
   # return jsonify({'Prediction':list(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
print(flask.__version__)