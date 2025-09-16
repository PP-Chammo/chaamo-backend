# openai_client.py
import os
from dotenv import load_dotenv
import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com")
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4), retry=retry_if_exception_type(httpx.HTTPError))
async def embed_texts(texts, model="text-embedding-3-small"):
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {"model": model, "input": texts}
        r = await client.post(f"{OPENAI_BASE}/v1/embeddings", headers=HEADERS, json=payload)
        r.raise_for_status()
        data = r.json()
        # data['data'] is list of embeddings
        return [item["embedding"] for item in data["data"]]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4), retry=retry_if_exception_type(httpx.HTTPError))
async def gpt_rerank(prompt: str, model="gpt-4o-mini", max_tokens=512):
    """
    Simple wrapper: sends a prompt and returns model text.
    For production, you might want to use responses API or chat api format.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "model": model,
            "messages": [{"role":"user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        r = await client.post(f"{OPENAI_BASE}/v1/chat/completions", headers=HEADERS, json=payload)
        r.raise_for_status()
        data = r.json()
        # extract assistant reply text
        return data["choices"][0]["message"]["content"]
