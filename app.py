import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

## import DecisionTreeRegressor model and standard scaler pickle
DecisionTreeRegressor_model=pickle.load(open(r'C:\Users\Gaurav\OneDrive\Desktop\rain_prediction\models\Dt_regg.pkl','rb'))
standard_scaler=pickle.load(open(r'C:\Users\Gaurav\OneDrive\Desktop\rain_prediction\models\scalers.pkl','rb'))

# Route for home page
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        maximum_temperature=float(request.form.get('tmax'))
        Minimum_temperature=float(request.form.get('tmin'))
        Wind_speed=float(request.form.get('wind_speed'))
        Dew_point=float(request.form.get('dew_point'))
        Relative_humidity=float(request.form.get('rel_humidity'))
        Air_pressure=float(request.form.get('press_kpa'))
        precipition=float(request.form.get('preci'))

        new_data_scaled=standard_scaler.transform([['maximum_temperature(C)','minimum_temperature(C)','Wind_speed','Dew_point','Relative_humidity ','Air_pressure(kPa)','precipition']])
        result=DecisionTreeRegressor_model.predict(new_data_scaled)
        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')
if __name__ == '__main__':
    app.run(host="0.0.0.0")