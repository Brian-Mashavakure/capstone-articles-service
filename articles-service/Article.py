from pydantic import BaseModel
from typing import Optional



class Article(BaseModel):
    subtitle: str
    title: str
    link: str
    displayed_link: str
    date: Optional[str] = None
    snippet: str
    snippet_highlighted_words: list[str]
    favicon: str
    source: str
    search_link: str
    serpapi_api_link: Optional[str] = None