from fastapi import FastAPI, HTTPException
import uvicorn
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os
from Article import Article

app = FastAPI()

load_dotenv()

apiKey = os.getenv("SERP_API_KEY")

def getcancerarticles():
    search = GoogleSearch({
        "q": "skin cancer",
        "api_key": apiKey
    })

    results = search.get_dict()  # Ensure results are returned as a dictionary
    articles = results.get('knowledge_graph', {}).get('buttons', [])

    for article in articles:
        article.setdefault('serpapi_api_link', '')
        article.setdefault('date', '')
    return articles


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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)


