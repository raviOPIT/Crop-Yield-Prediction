from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle
import pandas as pd
from pydantic import BaseModel

# Load the trained model (assumes model.pkl is in the same directory)
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    model = None

app = FastAPI()

# Mount static directory (for CSS/JS if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# For demonstration, define some allowed countries and items
COUNTRIES = ["United States", "India", "Brazil", "China", "Nigeria"]
ITEMS = ["Wheat", "Maize", "Rice", "Soybeans", "Barley"]

class PredictionInput(BaseModel):
    country: str
    item: str
    year: int
    average_rainfall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "countries": COUNTRIES, "items": ITEMS})

@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Create a DataFrame for the model
    df = pd.DataFrame([{
        "Country": input_data.country,
        "Item": input_data.item,
        "Year": input_data.year,
        "average_rainfall_mm_per_year": input_data.average_rainfall_mm_per_year,
        "pesticides_tonnes": input_data.pesticides_tonnes,
        "avg_temp": input_data.avg_temp
    }])
    
    # Make prediction
    # prediction = model.predict(df)
    # predicted_yield = prediction[0]
    predicted_yield = 1000.00
    
    return {"predicted_yield": predicted_yield}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
