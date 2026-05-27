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
const resultsTime   = document.getElementById("results-time");
const grid          = document.getElementById("grid");
const spinner       = document.getElementById("spinner");
const loadMoreWrap  = document.getElementById("load-more-wrap");
const loadMoreBtn   = document.getElementById("load-more-btn");

const PAGE_SIZE = 12;
let allResults = [];
let displayedCount = 0;

function showSpinner(msg) {
  spinner.querySelector(".spinner-text").textContent = msg || "Searching...";
  spinner.classList.remove("hidden");
}

function hideSpinner() { spinner.classList.add("hidden"); }

function setStatus(msg, type = "") {
  statusEl.textContent = msg;
  statusEl.className = `status${type ? " " + type : ""}`;
  statusEl.classList.remove("hidden");
}

function clearStatus() { statusEl.className = "status hidden"; }

function clearResults() {
  grid.innerHTML = "";
  resultsMeta.classList.add("hidden");
  loadMoreWrap.classList.add("hidden");
  allResults = [];
  displayedCount = 0;
  hideSpinner();
  clearStatus();
}

function renderCard({ image_id, image_url, score, caption }) {
  const pct = Math.round(score * 100);
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `
    <img src="${image_url}" alt="${image_id}" loading="lazy" />
    <div class="card-body">
      <div class="card-score">
        <div class="score-bar-wrap">
          <div class="score-bar" style="width:${pct}%"></div>
        </div>
        <span class="score-label">${pct}%</span>
      </div>
      <div class="card-caption">${caption || ""}</div>
    </div>`;
  card.addEventListener("click", () => openModal(image_url, caption));
  return card;
}

function showResults(data, elapsed) {
  hideSpinner();
  clearStatus();

  allResults = data.results;
  displayedCount = 0;

  resultsCount.textContent = `${Math.min(PAGE_SIZE, data.results.length)} of ${data.total}`;
  resultsQuery.textContent = data.query;
  resultsTime.textContent = elapsed ? `(${(elapsed / 1000).toFixed(1)}s)` : "";
  resultsMeta.classList.remove("hidden");

  if (data.results.length === 0) {
    setStatus("No results found.");
    return;
  }

  loadPage();
}

function loadPage() {
  const end = Math.min(displayedCount + PAGE_SIZE, allResults.length);
  for (let i = displayedCount; i < end; i++) {
    grid.appendChild(renderCard(allResults[i]));
  }
  displayedCount = end;
  resultsCount.textContent = `${displayedCount} of ${allResults.length}`;

  if (displayedCount < allResults.length) {
    loadMoreWrap.classList.remove("hidden");
  } else {
    loadMoreWrap.classList.add("hidden");
  }
}

loadMoreBtn.addEventListener("click", loadPage);

// ── Modal ────────────────────────────────────────────────
const modal        = document.getElementById("modal");
const modalImg     = document.getElementById("modal-img");
const modalCaption = document.getElementById("modal-caption");

function openModal(src, caption) {
  modalImg.src = src;
  modalCaption.textContent = caption || "";
  modal.classList.remove("hidden");
}

function closeModal() {
  modal.classList.add("hidden");
  modalImg.src = "";
}

document.getElementById("modal-close").addEventListener("click", closeModal);
document.querySelector(".modal-backdrop").addEventListener("click", closeModal);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

// ── Drop zone setup ──────────────────────────────────────
function setupDropZone(dropId, fileInputId, previewId, placeholderId, clearBtnId, onFile, onClear) {
  const drop        = document.getElementById(dropId);
  const fileInput   = document.getElementById(fileInputId);
  const preview     = document.getElementById(previewId);
  const placeholder = document.getElementById(placeholderId);
  const clearBtn    = document.getElementById(clearBtnId);

  function loadFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.classList.remove("hidden");
    placeholder.classList.add("hidden");
    clearBtn.classList.remove("hidden");
    onFile(file);
  }

  function reset() {
    preview.src = "";
    preview.classList.add("hidden");
    placeholder.classList.remove("hidden");
    clearBtn.classList.add("hidden");
    fileInput.value = "";
    onClear();
  }

  clearBtn.addEventListener("click", (e) => { e.stopPropagation(); reset(); });

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

setupDropZone("search-drop", "search-file", "search-preview", "search-placeholder", "search-clear", (f) => {
  searchFile = f;
  imageBtnEl.disabled = false;
}, () => {
  searchFile = null;
  imageBtnEl.disabled = true;
});

imageBtnEl.addEventListener("click", async () => {
  if (!searchFile) return;
  imageBtnEl.disabled = true;
  clearResults();
  showSpinner("Searching for similar images...");

  const form = new FormData();
  form.append("image", searchFile);
  form.append("top_k", 36);

  const t0 = Date.now();
  try {
    const res = await fetch("/api/search/image", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    showResults(await res.json(), Date.now() - t0);
  } catch (err) {
    hideSpinner();
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
  showSpinner("Searching...");

  const t0 = Date.now();
  try {
    const res = await fetch("/api/search/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 36 }),
    });
    if (!res.ok) throw new Error(await res.text());
    showResults(await res.json(), Date.now() - t0);
  } catch (err) {
    hideSpinner();
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

setupDropZone("upload-drop", "upload-file", "upload-preview", "upload-placeholder", "upload-clear", (f) => {
  uploadFile = f;
  uploadBtnEl.disabled = false;
}, () => {
  uploadFile = null;
  uploadBtnEl.disabled = true;
});

uploadBtnEl.addEventListener("click", async () => {
  if (!uploadFile) return;
  const caption = document.getElementById("upload-caption").value.trim();
  if (!caption) { setStatus("Please add a caption before uploading.", "error"); return; }

  uploadBtnEl.disabled = true;
  showSpinner("Uploading...");

  const form = new FormData();
  form.append("image", uploadFile);
  form.append("caption", caption);

  try {
    const res = await fetch("/api/upload/image", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    hideSpinner();
    setStatus(`Uploaded "${data.image_id}" successfully.`, "success");
    document.getElementById("upload-caption").value = "";
  } catch (err) {
    hideSpinner();
    setStatus(`Upload failed: ${err.message}`, "error");
  } finally {
    uploadBtnEl.disabled = false;
  }
});
