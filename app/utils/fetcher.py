import httpx
import urllib.parse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

async def fetch_html(query: str, region: str, page: str) -> str:
    """
    Fetches HTML content from eBay URL based on query and region.

    Args:
        query (str): Search keyword.
        region (str): Market region (e.g., 'uk', 'us').
        page (str): Market region (e.g., '1', '2', '3', '4', '5').

    Returns:
        str: HTML content of the page as text.

    Raises:
        httpx.HTTPStatusError: If response status code is not 2xx.
    """

    params = {
        "_from": "R40",
        "_nkw": query,
        "_sacat": "0",
        "rt": "nc",
        "LH_Sold": "1",
        "LH_Complete": "1",
        "Country/Region of Manufacture": "United States" if region == "us" else "United Kingdom",
        "_pgn": 1 if not page else page
    }
    domain = "com" if region == "us" else "co.uk"
    url = f"https://www.ebay.{domain}/sch/i.html?${urllib.parse.urlencode(params)}"

    print(f"Fetching URL: {url}")

    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, http2=True) as client:
        try:
            response = await client.get(url, timeout=20.0)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
            raise
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            raise

