(() => {
  // --- Tab switching ---
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".panel");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      panels.forEach((p) => { p.classList.remove("active"); p.classList.add("hidden"); });
      tab.classList.add("active");
      const panel = document.getElementById(`panel-${tab.dataset.mode}`);
      panel.classList.remove("hidden");
      panel.classList.add("active");
    });
  });

  // --- Image preview ---
  document.getElementById("image-file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const wrapper = document.getElementById("image-preview-wrapper");
    const preview = document.getElementById("image-preview");
    preview.src = URL.createObjectURL(file);
    wrapper.classList.remove("hidden");
  });

  // --- Helpers ---
  function showStatus(msg) {
    const bar = document.getElementById("status-bar");
    document.getElementById("status-msg").textContent = msg;
    bar.classList.remove("hidden");
  }
  function hideStatus() {
    document.getElementById("status-bar").classList.add("hidden");
  }

  function renderResults(data) {
    const meta = document.getElementById("results-meta");
    const grid = document.getElementById("results-grid");

    document.getElementById("results-count").textContent = data.total;
    document.getElementById("results-query").textContent = data.query;
    document.getElementById("results-time").textContent = data.took_ms;
    meta.classList.remove("hidden");

    if (data.results.length === 0) {
      grid.innerHTML = `<div class="no-results">No results found. Is the index built?</div>`;
      return;
    }

    grid.innerHTML = data.results.map((r) => `
      <div class="result-card">
        ${r.image_path ? `<img src="${r.image_path}" alt="${r.title || "result"}" loading="lazy" />` : ""}
        ${r.title ? `<div class="title">${r.title}</div>` : ""}
        ${r.text ? `<div class="excerpt">${r.text.slice(0, 200)}${r.text.length > 200 ? "…" : ""}</div>` : ""}
        <div class="score">Score: ${r.score.toFixed(4)} &nbsp;|&nbsp; ID: ${r.id}</div>
      </div>
    `).join("");
  }

  function handleError(err) {
    showStatus(`Error: ${err.message || "Something went wrong"}`);
  }

  // --- Text search ---
  document.getElementById("text-search-btn").addEventListener("click", async () => {
    const query = document.getElementById("text-query").value.trim();
    const top_k = parseInt(document.getElementById("text-topk").value, 10);
    if (!query) { showStatus("Please enter a query."); return; }

    hideStatus();
    showStatus("Searching…");
    try {
      const res = await fetch("/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k }),
      });
      if (!res.ok) throw new Error(await res.text());
      renderResults(await res.json());
      hideStatus();
    } catch (err) { handleError(err); }
  });

  // --- Image search ---
  document.getElementById("image-search-btn").addEventListener("click", async () => {
    const query = document.getElementById("image-query").value.trim();
    const file = document.getElementById("image-file").files[0];
    const top_k = document.getElementById("image-topk").value;
    if (!query && !file) { showStatus("Provide a text query or an image."); return; }

    hideStatus();
    showStatus("Searching…");
    const form = new FormData();
    if (query) form.append("query", query);
    if (file) form.append("image", file);
    form.append("top_k", top_k);

    try {
      const res = await fetch("/api/search/image", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      renderResults(await res.json());
      hideStatus();
    } catch (err) { handleError(err); }
  });

  // --- Multimodal search (reuses image endpoint) ---
  document.getElementById("multi-search-btn").addEventListener("click", async () => {
    const query = document.getElementById("multi-query").value.trim();
    const file = document.getElementById("multi-file").files[0];
    const top_k = document.getElementById("multi-topk").value;
    if (!query && !file) { showStatus("Provide at least a text query."); return; }

    hideStatus();
    showStatus("Searching…");
    const form = new FormData();
    if (query) form.append("query", query);
    if (file) form.append("image", file);
    form.append("top_k", top_k);

    try {
      const res = await fetch("/api/search/image", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      renderResults(await res.json());
      hideStatus();
    } catch (err) { handleError(err); }
  });

  // Enter key on text input
  document.getElementById("text-query").addEventListener("keydown", (e) => {
    if (e.key === "Enter") document.getElementById("text-search-btn").click();
  });
})();
