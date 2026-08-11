import flask
from flask import Flask
from flask import request,render_template
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import LinearRegression


with open("SLR_Model.pkl","rb") as t:
    reg = pickle.load(t)

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template("index.html")

@app.route("/predict",methods = ['GET','POST'])
def fun3():
    a = [float(i) for i in request.form.values()] # '12' -> [12.0]
    f = np.array([a])   #  np.array([[12.0])  -> 2D
    sol = reg.predict(f)[0][0]  # [[15.55]] -> sol[0] -> [1.55] -> sol[0][0] -> 15.55
    return render_template("index.html",prediction_text = sol)


if __name__ == "__main__":
    app.run(debug=True)  # link : http://127.0.0.1:5000/