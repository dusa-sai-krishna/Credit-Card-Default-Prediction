# Credit Card Default Prediction

This repository contains an end-to-end machine learning application that predicts whether a credit card customer will default on their next month's payment. The prediction model uses demographic factors, credit data, history of payments, and bill statements.

## About Dataset

### Overview
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

### Content
The dataset includes the following 25 variables:

- **ID**: ID of each client
- **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
- **SEX**: Gender (1=male, 2=female)
- **EDUCATION**: Education level (1=graduate school, 2=university, 3=high school, 4=others)
- **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
- **AGE**: Age in years
- **PAY_0**: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
- **PAY_2**: Repayment status in August, 2005 (scale same as above)
- **PAY_3**: Repayment status in July, 2005 (scale same as above)
- **PAY_4**: Repayment status in June, 2005 (scale same as above)
- **PAY_5**: Repayment status in May, 2005 (scale same as above)
- **PAY_6**: Repayment status in April, 2005 (scale same as above)
- **BILL_AMT1**: Amount of bill statement in September, 2005 (NT dollar)
- **BILL_AMT2**: Amount of bill statement in August, 2005 (NT dollar)
- **BILL_AMT3**: Amount of bill statement in July, 2005 (NT dollar)
- **BILL_AMT4**: Amount of bill statement in June, 2005 (NT dollar)
- **BILL_AMT5**: Amount of bill statement in May, 2005 (NT dollar)
- **BILL_AMT6**: Amount of bill statement in April, 2005 (NT dollar)
- **PAY_AMT1**: Amount of previous payment in September, 2005 (NT dollar)
- **PAY_AMT2**: Amount of previous payment in August, 2005 (NT dollar)
- **PAY_AMT3**: Amount of previous payment in July, 2005 (NT dollar)
- **PAY_AMT4**: Amount of previous payment in June, 2005 (NT dollar)
- **PAY_AMT5**: Amount of previous payment in May, 2005 (NT dollar)
- **PAY_AMT6**: Amount of previous payment in April, 2005 (NT dollar)
- **default.payment.next.month**: Default payment (1=yes, 0=no)


[DataSet Link](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## Project Structure

The project follows a modular structure to ensure scalability and maintainability. Below is an overview of the project structure:
```
├── .ebextensions
├── src
│ ├── components
│ │ ├── data_ingestion.py
│ │ ├── data_preprocessing.py
│ ├── pipelines
│ │ ├── prediction_pipeline.py
│ │ ├── training_pipeline.py
│ ├── logger.py
│ ├── exception.py
│ └── utils.py
├── notebooks
│ ├── EDA.ipynb
│ ├── Model_Training.ipynb
├── artifacts
│ ├── model.pkl
│ ├── preprocessor.pkl
├── app.py
├── templates
│ ├── index.html
├── application.py
├── requirements.txt
├── README.md
└── .gitignore
```


## Key Components

### Data Ingestion
The `data_ingestion.py` module is responsible for loading the dataset and performing initial preprocessing steps such as handling missing values and splitting the data into training and testing sets.

### Data Preprocessing
The `data_preprocessing.py` module includes functions to clean and transform the data. It utilizes pipelines to automate the preprocessing steps, including scaling and encoding.

### Model Training
The `Model_Training.ipynb` notebook contains the code for training the model on Google Colab due to the dataset size. The trained model is then saved as a pickle file (`model.pkl`) and placed in the `artifacts` folder.

### Pipelines
The project uses pipelines to streamline data processing and model training:
- `prediction_pipeline.py`: Automates the data preprocessing steps.
- `training_pipeline.py`: Automates the data ingestion, data preprocessing, model training, and evaluation process.

### Flask App
The `app.py` script creates an interactive front end using Flask, allowing users to input customer details and get the prediction.

## Technology Used
- **Python 3.7**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Flask**
- **Jupyter Notebook**
- **Google Colab**

## How to Run

### Prerequisites
Ensure you have Python 3.7 installed. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Steps to Run

Run the Training Pipeline: This step creates the pre-processor object.

```bash
python src/pipelines/training_pipeline.py

```
Train the Model: Open and run the Model_Training.ipynb file in Google Colab. After training, download the model.pkl file and place it in the artifacts folder.

Run the Flask App:

```bash
python app.py
```

Access the web application by navigating to http://localhost:5000 in your web browser.