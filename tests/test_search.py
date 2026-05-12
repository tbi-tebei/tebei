import pytest


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_text_search_empty_index(client):
    response = await client.post("/api/search/text", json={"query": "hello world", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "hello world"
    assert data["total"] == 0
    assert isinstance(data["results"], list)


async def test_image_search_no_input(client):
    response = await client.post("/api/search/image")
    assert response.status_code == 422  # missing required fields


async def test_index_status(client):
    response = await client.get("/api/index/status")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "docs_indexed" in data
