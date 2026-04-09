from typing import Literal
from pydantic import BaseModel, Field

class ArticleSummary(BaseModel):
    headline: str = Field(..., description="Article headline")
    source: str = Field(..., description="Publisher/source name")
    date: str = Field(..., description="YYYY-MM-DD date or 'N/A'")
    link: str = Field(..., description="Source URL or 'N/A'")
    summary: str = Field(..., description="1-3 sentence summary tailored to the query")
    relevance: str = Field(..., description="Short clause explaining why this article is relevant to the query")

class UnifiedResponse(BaseModel):
    type: Literal["news", "financial"] = Field(..., description="Whether the response is news-focused or financial analysis")
    query: str = Field(..., description="The original user query")
    answer: str = Field(..., description="2-4 sentence synthesized answer that directly addresses the query")
    top_articles: list[ArticleSummary] = Field(
        default_factory=list,
        description="At least 5 most relevant articles (minimum 5, include more if available); use [] if no sources were used",
    )
