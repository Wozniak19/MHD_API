import pandas as pd
import uvicorn
from fastapi import FastAPI
import joblib

app = FastAPI()
model  = joblib.load('GBMClassifier_model.pkl')

@app.get('/')
async def hello():
    return "HELLO WORLD"


@app.post("/classify")
async def read_root(input:dict):
    
    
    df = pd.DataFrame(input,index=range(1))

    predicted =  model.predict(df.values)
    Depression = str(predicted[0][0])
    Schizophrenia = str(predicted[0][1])
    Acute_and_transient_psychotic_disorder = str(predicted[0][2])
    Delusional_Disorder = str(predicted[0][3])
    BiPolar1 = str(predicted[0][4])
    BiPolar2 = str(predicted[0][5])
    Anxiety = str(predicted[0][6])
    Generalized_Anxiety = str(predicted[0][7])
    Panic_Disorder = str(predicted[0][8])
    Specific_Phobia = str(predicted[0][9])
    Social_Anxiety = str(predicted[0][10])
    OCD= str(predicted[0][11])
    PTSD= str(predicted[0][12])
    Gambling = str(predicted[0][13])
    substance_abuse = str(predicted[0][14])
    


    return {
        "Depression":Depression,
        "Schizophrenia":Schizophrenia,
        "Acute_and_transient_psychotic_disorder":Acute_and_transient_psychotic_disorder,
        "Delusional_Disorder":Delusional_Disorder,
        "BiPolar1":BiPolar1,
        "BiPolar2":BiPolar2,
        "Anxiety":Anxiety,
        "Generalized_Anxiety":Generalized_Anxiety,
        "Panic_Disorder":Panic_Disorder,
        "Specific_Phobia":Specific_Phobia,
        "Social_Anxiety":Social_Anxiety,
        "OCD":OCD,
        "PTSD": PTSD,
        "Gambling":Gambling,
        "substance_abuse":substance_abuse

    }
    























if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)