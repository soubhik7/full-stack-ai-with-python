const searchInput = document.getElementById("search-input");
const suggestionsEl = document.getElementById("suggestions");
const selectedChipsEl = document.getElementById("selected-chips");
const selectedCountEl = document.getElementById("selected-count");
const fetchBtn = document.getElementById("fetch-btn");
const statusEl = document.getElementById("status");
const resultsBox = document.getElementById("results-box");
const downloadLink = document.getElementById("download-link");
const resultsTable = document.getElementById("results-table");

const selected = new Map(); // trading_symbol -> {name, exchange}
let debounceTimer = null;

function debounce(fn, delay) {
  return (...args) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => fn(...args), delay);
  };
}

async function runSearch(query) {
  if (!query.trim()) {
    suggestionsEl.innerHTML = "";
    return;
  }
  const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
  const items = await res.json();
  renderSuggestions(items);
}

function renderSuggestions(items) {
  suggestionsEl.innerHTML = "";
  if (items.length === 0) {
    suggestionsEl.innerHTML = "<li class='empty'>No matches</li>";
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="name">${item.name}</span> <span class="symbol">${item.trading_symbol} · ${item.exchange}</span>`;
    li.addEventListener("click", () => addSelected(item));
    suggestionsEl.appendChild(li);
  }
}

function addSelected(item) {
  if (selected.has(item.trading_symbol)) return;
  selected.set(item.trading_symbol, item);
  renderSelected();
  searchInput.value = "";
  suggestionsEl.innerHTML = "";
  searchInput.focus();
}

function removeSelected(symbol) {
  selected.delete(symbol);
  renderSelected();
}

function renderSelected() {
  selectedChipsEl.innerHTML = "";
  for (const [symbol, item] of selected) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `${item.name} (${symbol}) <button aria-label="remove">&times;</button>`;
    chip.querySelector("button").addEventListener("click", () => removeSelected(symbol));
    selectedChipsEl.appendChild(chip);
  }
  selectedCountEl.textContent = selected.size;
  fetchBtn.disabled = selected.size === 0;
}

async function fetchData() {
  statusEl.textContent = "Fetching...";
  fetchBtn.disabled = true;
  try {
    const res = await fetch("/api/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbols: Array.from(selected.keys()) }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Fetch failed");
    }
    const data = await res.json();
    statusEl.textContent = `Saved ${data.rows.length} companies to ${data.filename}`;
    downloadLink.href = `/api/download/${data.filename}`;
    renderTable(data.rows);
    resultsBox.hidden = false;
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    fetchBtn.disabled = selected.size === 0;
  }
}

function renderTable(rows) {
  const thead = resultsTable.querySelector("thead");
  const tbody = resultsTable.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";
  if (rows.length === 0) return;

  const columns = Object.keys(rows[0]);
  const headRow = document.createElement("tr");
  for (const col of columns) {
    const th = document.createElement("th");
    th.textContent = col;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);

  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const col of columns) {
      const td = document.createElement("td");
      td.textContent = row[col] ?? "";
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

searchInput.addEventListener("input", debounce((e) => runSearch(e.target.value), 250));
fetchBtn.addEventListener("click", fetchData);
