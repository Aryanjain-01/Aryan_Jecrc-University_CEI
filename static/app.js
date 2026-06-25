const documentList = document.querySelector("#documentList");
const statsEl = document.querySelector("#stats");
const uploadForm = document.querySelector("#uploadForm");
const uploadStatus = document.querySelector("#uploadStatus");
const fileInput = document.querySelector("#fileInput");
const reportText = document.querySelector("#reportText");
const queryForm = document.querySelector("#queryForm");
const queryInput = document.querySelector("#queryInput");
const answerEl = document.querySelector("#answer");
const metricsEl = document.querySelector("#metrics");
const sourcesEl = document.querySelector("#sources");
const positivesEl = document.querySelector("#positives");
const expenditureEl = document.querySelector("#expenditure");
const loadSampleBtn = document.querySelector("#loadSampleBtn");
const clearBtn = document.querySelector("#clearBtn");

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || "Request failed");
  }
  return payload;
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function renderDocuments(documents) {
  if (!documents.length) {
    documentList.innerHTML = `<div class="document"><strong>No reports indexed</strong><span>Add a report or load the sample set.</span></div>`;
    return;
  }

  documentList.innerHTML = documents
    .map(
      (document) => `
        <div class="document">
          <strong>${document.name}</strong>
          <span>${document.characters.toLocaleString()} characters</span>
        </div>
      `,
    )
    .join("");
}

function renderAnswer(payload) {
  answerEl.classList.remove("empty");
  answerEl.textContent = payload.answer;

  metricsEl.innerHTML = payload.metrics.length
    ? payload.metrics.map((metric) => `<span class="chip">${metric}</span>`).join("")
    : `<span class="empty">No finance metrics detected in the retrieved chunks.</span>`;

  sourcesEl.innerHTML = payload.sources
    .map(
      (source) => `
        <article class="source">
          <div class="source-header">
            <span>${source.document_name} · chunk ${source.chunk_index}</span>
            <span>${Math.round(source.score * 100)}%</span>
          </div>
          <p>${source.preview}</p>
        </article>
      `,
    )
    .join("");
}

function renderInsights(insights) {
  // Render Positives
  if (insights.positives && insights.positives.length > 0) {
    positivesEl.innerHTML = insights.positives
      .map((positive) => `<div class="insight-item positive-item">✓ ${positive}</div>`)
      .join("");
  } else {
    positivesEl.innerHTML = `<div class="insight-item empty">No company strengths detected yet.</div>`;
  }

  // Render Expenditure Breakdown
  if (insights.expenditure && Object.keys(insights.expenditure).length > 0) {
    let expenditureHTML = "";
    for (const [category, items] of Object.entries(insights.expenditure)) {
      expenditureHTML += `
        <div class="expenditure-category">
          <div class="category-name">💰 ${category}</div>
          <ul class="category-items">
            ${items.map((item) => `<li>${item}</li>`).join("")}
          </ul>
        </div>
      `;
    }
    expenditureEl.innerHTML = expenditureHTML;
  } else {
    expenditureEl.innerHTML = `<div class="insight-item empty">No expenditure details detected yet.</div>`;
  }
}

async function refresh() {
  const [documents, stats, insights] = await Promise.all([
    api("/api/documents"),
    api("/api/stats"),
    api("/api/insights"),
  ]);
  renderDocuments(documents);
  statsEl.textContent = `${stats.documents} documents · ${stats.chunks} chunks`;
  renderInsights(insights);
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  uploadStatus.textContent = "Indexing report...";

  try {
    const file = fileInput.files[0];
    const payload = {
      name: file ? file.name : "Pasted finance report",
      text: reportText.value,
    };

    if (file) {
      payload.fileBase64 = await fileToBase64(file);
    }

    await api("/api/documents", { method: "POST", body: JSON.stringify(payload) });
    uploadStatus.textContent = "Report indexed.";
    fileInput.value = "";
    reportText.value = "";
    await refresh();
  } catch (error) {
    uploadStatus.textContent = error.message;
  }
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  answerEl.textContent = "Retrieving evidence...";
  answerEl.classList.add("empty");
  metricsEl.innerHTML = "";
  sourcesEl.innerHTML = "";

  try {
    const payload = await api("/api/query", {
      method: "POST",
      body: JSON.stringify({ query: queryInput.value, topK: 5 }),
    });
    renderAnswer(payload);
  } catch (error) {
    answerEl.textContent = error.message;
  }
});

loadSampleBtn.addEventListener("click", async () => {
  await api("/api/load-sample", { method: "POST", body: JSON.stringify({}) });
  await refresh();
});

clearBtn.addEventListener("click", async () => {
  await api("/api/clear", { method: "POST", body: JSON.stringify({}) });
  answerEl.textContent = "Library cleared. Add reports to start again.";
  answerEl.classList.add("empty");
  metricsEl.innerHTML = "";
  sourcesEl.innerHTML = "";
  await refresh();
});

refresh().catch((error) => {
  statsEl.textContent = error.message;
});
