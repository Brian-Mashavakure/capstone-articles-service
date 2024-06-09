from fastapi import FastAPI, HTTPException, File, UploadFile, Request
import uvicorn
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import io


from Article import Article


app = FastAPI()

load_dotenv()


#articles functionality
apiKey = os.getenv("SERP_API_KEY")

def getcancerarticles():
    search = GoogleSearch({
        "q": "skin cancer",
        "api_key": apiKey
    })

    results = search.get_dict()
    print(results)
    articles = results.get('knowledge_graph', {}).get('buttons', [])

    for article in articles:
        article.setdefault('serpapi_api_link', '')
        article.setdefault('date', '')
    print(articles)
    return articles


#model functionality
# Load the pre-trained model
model = tf.keras.models.load_model("model/my_model.h5")

# Define labels
labels = {0: "malignant", 1: "benign"}


@app.get('/capstone/api/articles')
def index()->list[Article]:
    try:
        articles = getcancerarticles()
        if articles:
            return articles
        else:
            raise HTTPException(status_code=404, detail="No articles found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/capstone/api/model')
async def predict(request: Request):
    try:
        # Read raw request body
        img_bytes = await request.body()

        # Convert bytes to PIL image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Resize image to match model input shape
        img = img.resize((128, 128))
        # Convert image to numpy array
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        return {"prediction": labels[predicted_class]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)


