const queryInput = document.getElementById("query");
const searchBtn = document.getElementById("search-btn");
const status = document.getElementById("status");
const grid = document.getElementById("grid");
const resultsMeta = document.getElementById("results-meta");
const resultsCount = document.getElementById("results-count");
const resultsQuery = document.getElementById("results-query");

async function search() {
  const query = queryInput.value.trim();
  if (!query) return;

  searchBtn.disabled = true;
  grid.innerHTML = "";
  resultsMeta.classList.add("hidden");
  status.textContent = "Searching...";
  status.classList.remove("hidden");

  try {
    const res = await fetch("/api/search/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 12 }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    status.classList.add("hidden");

    resultsCount.textContent = data.total;
    resultsQuery.textContent = data.query;
    resultsMeta.classList.remove("hidden");

    if (data.results.length === 0) {
      status.textContent = "No results found.";
      status.classList.remove("hidden");
      return;
    }

    grid.innerHTML = data.results.map(({ image_id, image_url }) => `
      <div class="card">
        <img src="${image_url}" alt="${image_id}" loading="lazy" />
        <div class="card-label">${image_id}</div>
      </div>
    `).join("");

  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    status.classList.remove("hidden");
  } finally {
    searchBtn.disabled = false;
  }
}

searchBtn.addEventListener("click", search);
queryInput.addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
