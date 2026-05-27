import pytest


async def test_health(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_home(client):
    response = await client.get("/")
    assert response.status_code == 200


async def test_text_search_returns_results(client):
    response = await client.post("/api/search/text", json={"query": "dog", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] > 0
    assert len(data["results"]) > 0


async def test_text_search_result_structure(client):
    response = await client.post("/api/search/text", json={"query": "beach", "top_k": 3})
    assert response.status_code == 200
    results = response.json()["results"]
    for r in results:
        assert "image_id" in r
        assert "image_url" in r
        assert "score" in r
        assert "caption" in r


async def test_text_search_respects_top_k(client):
    response = await client.post("/api/search/text", json={"query": "sunset", "top_k": 5})
    assert response.status_code == 200
    assert len(response.json()["results"]) <= 5


async def test_text_search_empty_query(client):
    response = await client.post("/api/search/text", json={"query": "", "top_k": 5})
    assert response.status_code == 200


async def test_text_search_indonesian_query(client):
    response = await client.post("/api/search/text", json={"query": "pantai", "top_k": 5})
    assert response.status_code == 200
    assert response.json()["total"] > 0


async def test_image_search_no_file(client):
    response = await client.post("/api/search/image")
    assert response.status_code == 422