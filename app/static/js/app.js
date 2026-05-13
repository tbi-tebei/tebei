// ── Tab switching ────────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");
    clearResults();
  });
});

// ── Shared helpers ───────────────────────────────────────
const statusEl      = document.getElementById("status");
const resultsMeta   = document.getElementById("results-meta");
const resultsCount  = document.getElementById("results-count");
const resultsQuery  = document.getElementById("results-query");
const grid          = document.getElementById("grid");

function setStatus(msg, type = "") {
  statusEl.textContent = msg;
  statusEl.className = `status${type ? " " + type : ""}`;
  statusEl.classList.remove("hidden");
}

function clearStatus() { statusEl.className = "status hidden"; }

function clearResults() {
  grid.innerHTML = "";
  resultsMeta.classList.add("hidden");
  clearStatus();
}

function renderResults(data) {
  clearStatus();
  resultsCount.textContent = data.total;
  resultsQuery.textContent = data.query;
  resultsMeta.classList.remove("hidden");

  if (data.results.length === 0) {
    setStatus("No results found.");
    return;
  }

  grid.innerHTML = data.results.map(({ image_id, image_url, score, caption }) => {
    const pct = Math.round(score * 100);
    return `
      <div class="card">
        <img src="${image_url}" alt="${image_id}" loading="lazy" />
        <div class="card-body">
          <div class="card-score">
            <div class="score-bar-wrap">
              <div class="score-bar" style="width:${pct}%"></div>
            </div>
            <span class="score-label">${pct}%</span>
          </div>
          <div class="card-caption">${caption || "—"}</div>
        </div>
      </div>`;
  }).join("");
}

// ── Drop zone setup ──────────────────────────────────────
function setupDropZone(dropId, fileInputId, previewId, placeholderId, onFile) {
  const drop        = document.getElementById(dropId);
  const fileInput   = document.getElementById(fileInputId);
  const preview     = document.getElementById(previewId);
  const placeholder = document.getElementById(placeholderId);

  function loadFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.classList.remove("hidden");
    placeholder.classList.add("hidden");
    onFile(file);
  }

  drop.addEventListener("dragover",  (e) => { e.preventDefault(); drop.classList.add("drag-over"); });
  drop.addEventListener("dragleave", ()  => drop.classList.remove("drag-over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("drag-over");
    loadFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener("change", () => loadFile(fileInput.files[0]));
}

// ── Image Search ─────────────────────────────────────────
let searchFile = null;
const imageBtnEl = document.getElementById("image-btn");

setupDropZone("search-drop", "search-file", "search-preview", "search-placeholder", (f) => {
  searchFile = f;
  imageBtnEl.disabled = false;
});

imageBtnEl.addEventListener("click", async () => {
  if (!searchFile) return;
  imageBtnEl.disabled = true;
  clearResults();
  setStatus("Searching for similar images...");

  const form = new FormData();
  form.append("image", searchFile);
  form.append("top_k", 12);

  try {
    const res = await fetch("/api/search/image", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    renderResults(await res.json());
  } catch (err) {
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    imageBtnEl.disabled = false;
  }
});

// ── Text Search ──────────────────────────────────────────
const textBtnEl   = document.getElementById("text-btn");
const textQueryEl = document.getElementById("text-query");

async function runTextSearch() {
  const query = textQueryEl.value.trim();
  if (!query) return;
  textBtnEl.disabled = true;
  clearResults();
  setStatus("Searching...");

  try {
    const res = await fetch("/api/search/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 12 }),
    });
    if (!res.ok) throw new Error(await res.text());
    renderResults(await res.json());
  } catch (err) {
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    textBtnEl.disabled = false;
  }
}

textBtnEl.addEventListener("click", runTextSearch);
textQueryEl.addEventListener("keydown", (e) => { if (e.key === "Enter") runTextSearch(); });

// ── Upload ───────────────────────────────────────────────
let uploadFile = null;
const uploadBtnEl = document.getElementById("upload-btn");

setupDropZone("upload-drop", "upload-file", "upload-preview", "upload-placeholder", (f) => {
  uploadFile = f;
  uploadBtnEl.disabled = false;
});

uploadBtnEl.addEventListener("click", async () => {
  if (!uploadFile) return;
  const caption = document.getElementById("upload-caption").value.trim();
  if (!caption) { setStatus("Please add a caption before uploading.", "error"); return; }

  uploadBtnEl.disabled = true;
  setStatus("Uploading...");

  const form = new FormData();
  form.append("image", uploadFile);
  form.append("caption", caption);

  try {
    const res = await fetch("/api/upload/image", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    setStatus(`Uploaded "${data.image_id}" successfully.`, "success");
    document.getElementById("upload-caption").value = "";
  } catch (err) {
    setStatus(`Upload failed: ${err.message}`, "error");
  } finally {
    uploadBtnEl.disabled = false;
  }
});
