from fastapi import FastAPI, File, UploadFile
from io import StringIO
import pandas as pd
from joblib import load


app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict/")
async def predict_house(file: UploadFile = File(...)): #queda esperando el archivo en la solicitud
    classifier = load("linear_regression.joblib")  #carga el modelo previamente entrenado
    features_df = pd.read_csv("selected_features.csv")  #lee el archivo csv enviado
    features = features_df['0'].to_list()  #convierte a lista las caracteristicas seleccionadas
    content = await file.read()  #lee el archivo recibido en la api
    
    df = pd.read_csv(StringIO(content.decode('utf-8')))  #convierte el contenido a un dataframe
    df = df[features]  #filtra el dataframe con las caracteristicas seleccionadas (features : columnas definidas)
    
    predictions = classifier.predict(df)  #realiza la prediccion
    return {"predictions": predictions.tolist()}  #devuelve las predicciones en formato lista
