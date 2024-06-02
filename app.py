

#import libraries
from src.pipelines.prediction_pipeline import PredictionPipeline,CustomData

from flask import Flask,render_template,url_for,redirect,request    

app = Flask(__name__)

@app.route('/')
def load_home():
    return render_template('index.html',prediction_text="")

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=="GET":
        return redirect(url_for('load_home'))
    else:
        ID = 0 # placeholder
        LIMIT_BAL = float(request.form['LIMIT_BAL'])
        SEX = request.form['SEX']
        EDUCATION = request.form['EDUCATION']
        MARRIAGE = request.form['MARRIAGE']
        AGE = int(request.form['AGE'])
        PAY_0 = int(request.form['REPAYMENT_STATUS_SEPT'])
        PAY_2 = int(request.form['REPAYMENT_STATUS_AUGUST'])
        PAY_3 = int(request.form['REPAYMENT_STATUS_JULY'])
        PAY_4 = int(request.form['REPAYMENT_STATUS_JUNE'])
        PAY_5 = int(request.form['REPAYMENT_STATUS_MAY'])
        PAY_6 = int(request.form['REPAYMENT_STATUS_APRIL'])
        BILL_AMT1 = float(request.form['BILL_AMT_SEPT'])
        BILL_AMT2 = float(request.form['BILL_AMT_AUGUST'])
        BILL_AMT3 = float(request.form['BILL_AMT_JULY'])
        BILL_AMT4 = float(request.form['BILL_AMT_JUNE'])
        BILL_AMT5 = float(request.form['BILL_AMT_MAY'])
        BILL_AMT6 = float(request.form['BILL_AMT_APRIL'])
        PAY_AMT1 = float(request.form['PAY_AMT_SEPT'])
        PAY_AMT2 = float(request.form['PAY_AMT_AUGUST'])
        PAY_AMT3 = float(request.form['PAY_AMT_JULY'])
        PAY_AMT4 = float(request.form['PAY_AMT_JUNE'])
        PAY_AMT5 = float(request.form['PAY_AMT_MAY'])
        PAY_AMT6 = float(request.form['PAY_AMT_APRIL'])

        obj = CustomData(
                ID,
                LIMIT_BAL,
                SEX,
                EDUCATION,
                MARRIAGE,
                AGE,
                PAY_0,
                PAY_2,
                PAY_3,
                PAY_4,
                PAY_5,
                PAY_6,
                BILL_AMT1,
                BILL_AMT2,
                BILL_AMT3,
                BILL_AMT4,
                BILL_AMT5,
                BILL_AMT6,
                PAY_AMT1,
                PAY_AMT2,
                PAY_AMT3,
                PAY_AMT4,
                PAY_AMT5,
                PAY_AMT6,
            )

        df = obj.get_data_as_dataframe()
        
        #get prediction
        prediction_agent=PredictionPipeline()
        prediction=prediction_agent.predict(df)
        prediction=prediction[0]
        if prediction==0: msg="Will not Default"
        elif prediction==1: msg="Will Default"
        else: msg="Error in Prediction"
        return render_template('index.html',prediction_text=msg)

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True,port=5000)