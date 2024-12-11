import os
import gdown
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import joblib
import pandas as pd
from pydantic import BaseModel

# Load the trained model (assumes model.pkl is in the same directory)

# Function to download the file from Google Drive
def download_model_from_google_drive():
    try:
        url = "https://drive.google.com/uc?id=1FhTHdUunq6gemJ5LJUlMV3SagBI5ouWd"
        output = "voting.pkl"

        gdown.download(url, output, quiet=False)
    except:
        print("model download failed")

def download_encoder_from_google_drive():
    try:
        url = "https://drive.google.com/uc?id=1-xp8GnHMtUPsN1HNnbVYtjyBQiLZ4jNY"
        output = "one_hot_encoder.pkl"

        gdown.download(url, output, quiet=False)
    except:
        print("encoder download failed")

def download_scaler_from_google_drive():

    try:
        url = "https://drive.google.com/uc?id=1Oa8LokajpqE2wygzAtO98ENb8oZ6JrbE"
        output = "standard_scaler.pkl"

        gdown.download(url, output, quiet=False)
    except:
        print("scaler download failed")

filename_model = "voting.pkl"
filename_encoder = "one_hot_encoder.pkl"
filename_scaler = "standard_scaler.pkl"

# Check if the file exists, else download it
if not os.path.exists(filename_model):
    print("Downloading Model")
    download_model_from_google_drive()
# Check if the file exists, else download it
if not os.path.exists(filename_encoder):
    print("Downloading Encoder")
    download_encoder_from_google_drive()
# Check if the file exists, else download it
if not os.path.exists(filename_scaler):
    print("Downloading Scaler")
    download_scaler_from_google_drive()

# Load the file using joblib
try:
    model = joblib.load(filename_model)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    encoder = joblib.load('one_hot_encoder.pkl')
    print("encoder loaded successfully!")
except Exception as e:
    print(f"Error loading the encoder: {e}")

try:
    scaler = joblib.load('standard_scaler.pkl')
    print("scaler loaded successfully!")
except Exception as e:
    print(f"Error loading the scaler: {e}")



app = FastAPI()

# Mount static directory (for CSS/JS if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

categorical_features = ['Area', 'Item']
numerical_features = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

# For demonstration, define some allowed countries and items
AREAS = ['Albania',
 'Algeria',
 'Angola',
 'Argentina',
 'Armenia',
 'Australia',
 'Austria',
 'Azerbaijan',
 'Bahamas',
 'Bahrain',
 'Bangladesh',
 'Belarus',
 'Belgium',
 'Botswana',
 'Brazil',
 'Bulgaria',
 'Burkina Faso',
 'Burundi',
 'Cameroon',
 'Canada',
 'Central African Republic',
 'Chile',
 'Colombia',
 'Croatia',
 'Denmark',
 'Dominican Republic',
 'Ecuador',
 'Egypt',
 'El Salvador',
 'Eritrea',
 'Estonia',
 'Finland',
 'France',
 'Germany',
 'Ghana',
 'Greece',
 'Guatemala',
 'Guinea',
 'Guyana',
 'Haiti',
 'Honduras',
 'Hungary',
 'India',
 'Indonesia',
 'Iraq',
 'Ireland',
 'Italy',
 'Jamaica',
 'Japan',
 'Kazakhstan',
 'Kenya',
 'Latvia',
 'Lebanon',
 'Lesotho',
 'Libya',
 'Lithuania',
 'Madagascar',
 'Malawi',
 'Malaysia',
 'Mali',
 'Mauritania',
 'Mauritius',
 'Mexico',
 'Montenegro',
 'Morocco',
 'Mozambique',
 'Namibia',
 'Nepal',
 'Netherlands',
 'New Zealand',
 'Nicaragua',
 'Niger',
 'Norway',
 'Pakistan',
 'Papua New Guinea',
 'Peru',
 'Poland',
 'Portugal',
 'Qatar',
 'Romania',
 'Rwanda',
 'Saudi Arabia',
 'Senegal',
 'Slovenia',
 'South Africa',
 'Spain',
 'Sri Lanka',
 'Sudan',
 'Suriname',
 'Sweden',
 'Switzerland',
 'Tajikistan',
 'Thailand',
 'Tunisia',
 'Turkey',
 'Uganda',
 'Ukraine',
 'United Kingdom',
 'Uruguay',
 'Zambia',
 'Zimbabwe']

ITEMS = ['Maize',
 'Potatoes',
 'Rice, paddy',
 'Sorghum',
 'Soybeans',
 'Wheat',
 'Cassava',
 'Sweet potatoes',
 'Plantains and others',
 'Yams']

class PredictionInput(BaseModel):
    Area: str
    item: str
    year: int
    average_rainfall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "Areas": AREAS, "items": ITEMS})

@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Create a DataFrame for the model
    df = pd.DataFrame([{
        "Area": input_data.Area,
        "Item": input_data.item,
        "Year": input_data.year,
        "average_rain_fall_mm_per_year": input_data.average_rainfall_mm_per_year,
        "pesticides_tonnes": input_data.pesticides_tonnes,
        "avg_temp": input_data.avg_temp
    }])

    encoded_categories = encoder.transform(df[categorical_features])
    scaled_numericals = scaler.transform(df[numerical_features])

    # Combine all features into a single dataset
    X = pd.concat(
    [
        pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(categorical_features)),
        pd.DataFrame(scaled_numericals, columns=numerical_features),
    ],
    axis=1,
    )
    # Make prediction
    prediction = model.predict(X)
    predicted_yield = prediction[0]
    # predicted_yield = 1000.00
    
    return {"predicted_yield": predicted_yield}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
