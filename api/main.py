from fastapi import FastAPI , UploadFile , File
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL  = tf.keras.models.load_model("../models/my_model.keras")
class_names = ["Early_Blight","Late_Blight","Healthy"]

@app.get("/ping")
def ping():
    return "Hello I am Alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0) # it will convert 1 dim array to 2 dimension
    prediction = MODEL.predict(img_batch)
    class_name = class_names[np.argmax(prediction)]
    confidence = np.max(prediction[0])
    return class_name,confidence

if __name__ == '__main__':
    uvicorn.run(app,host = "localhost",port = 8000)
