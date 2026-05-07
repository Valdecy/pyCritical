/* ============================================================
 * pyCritical · Workbench front-end
 * ============================================================
 * Sections:
 *   1. State + DOM refs
 *   2. Examples
 *   3. Method switching + editor rendering
 *   4. Compute (server round-trip)
 *   5. KPI strip + tab visibility
 *   6. Gantt visualization (SVG)
 *   7. Network visualization (SVG)
 *   8. Probability visualization (SVG, PERT)
 *   9. Results table + CSV export
 *  10. Tooltip helper
 *  11. Bootstrap
 * ============================================================ */


/* ---------- 1. State ---------- */

const SVGNS = "http://www.w3.org/2000/svg";

const state = {
  method: "cpm",
  tasks: [],             // editor task list (shape varies by method)
  result: null,          // last server response
  view: "gantt",        // active visualization tab
  editorMode: "paste",  // paste | manual
  pasteDirty: false,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const refs = {
  methodTabs:      $$(".method-tab"),
  editorModeTabs:  $$(".editor-mode-tab"),
  vizTabs:         $$(".viz-tab"),
  panePaste:       $("#panePaste"),
  paneManual:      $("#paneManual"),
  datasetPaste:    $("#datasetPaste"),
  btnParsePaste:   $("#btnParsePaste"),
  pasteStatus:     $("#pasteStatus"),
  editorFootHint:  $("#editorFootHint"),
  taskThead:       $("#taskThead"),
  taskTbody:       $("#taskTbody"),
  panelHint:       $("#panelHint"),
  btnAdd:          $("#btnAddRow"),
  btnRun:          $("#btnRun"),
  btnClear:        $("#btnClear"),
  btnLoad:         $("#btnLoadExample"),
  btnLoad2:        $("#btnLoadExample2"),
  vizEmpty:        $("#vizEmpty"),
  vizError:        $("#vizError"),
  vizErrorMsg:     $("#vizErrorMsg"),
  paneGantt:       $("#paneGantt"),
  paneNetwork:     $("#paneNetwork"),
  paneProb:        $("#paneProbability"),
  paneTable:       $("#paneTable"),
  ganttWrap:       $("#ganttWrap"),
  ganttShowSlack:  $("#ganttShowSlack"),
  ganttShowDeps:   $("#ganttShowDeps"),
  networkWrap:     $("#networkWrap"),
  netRelayout:     $("#netRelayout"),
  probTabBtn:      $('.viz-tab[data-view="probability"]'),
  probDate:        $("#probDate"),
  probValue:       $("#probValue"),
  probMean:        $("#probMean"),
  probSigma:       $("#probSigma"),
  probZ:           $("#probZ"),
  probChartWrap:   $("#probChartWrap"),
  resultsTable:    $("#resultsTable"),
  btnDownloadCsv:  $("#btnDownloadCsv"),
  busyOverlay:     $("#busyOverlay"),
  busyTitle:       $("#busyTitle"),
  busyMsg:         $("#busyMsg"),
  kpiDuration:     $("#kpiDuration"),
  kpiStdDev:       $("#kpiStdDev"),
  kpiStdDevWrap:   $("#kpiStdDevWrap"),
  kpiTasks:        $("#kpiTasks"),
  kpiPath:         $("#kpiPath"),
  tooltip:         $("#tooltip"),
};


/* ---------- 2. Examples ---------- */
/* Examples are crafted to be small but illuminating: a couple of parallel
 * branches, at least one task with slack, and a clear critical path. */

const EXAMPLES = {
  cpm: [
    { id: "A", preds: [],           duration: 3 },
    { id: "B", preds: [],           duration: 5 },
    { id: "C", preds: ["A"],        duration: 4 },
    { id: "D", preds: ["A"],        duration: 3 },
    { id: "E", preds: ["B"],        duration: 7 },
    { id: "F", preds: ["B"],        duration: 7 },
    { id: "G", preds: ["C"],        duration: 7 },
    { id: "H", preds: ["D", "E"],   duration: 7 },
    { id: "I", preds: ["F"],        duration: 2 },
  ],
  pert: [
    { id: "A", preds: [],             o: 4,  m: 6,  p: 8 },
    { id: "B", preds: ["A"],          o: 3,  m: 4,  p: 6 },
    { id: "C", preds: [],             o: 2,  m: 3,  p: 6 },
    { id: "D", preds: [],             o: 7,  m: 11, p: 12 },
    { id: "E", preds: ["B", "C"],    o: 9,  m: 12, p: 15 },
    { id: "F", preds: ["B", "C"],    o: 22, m: 26, p: 27 },
    { id: "G", preds: ["A"],          o: 6,  m: 7,  p: 8 },
    { id: "H", preds: ["D", "E", "G"], o: 9, m: 12, p: 15 },
    { id: "I", preds: ["F", "H"],    o: 5,  m: 7,  p: 9 },
  ],
  cpm_dep: [
    { id: "A", preds: [], duration: 3 },
    { id: "B", preds: [], duration: 5 },
    { id: "C", preds: [{ id: "A", type: "FS", lag: 0 }], duration: 4 },
    { id: "D", preds: [{ id: "A", type: "FS", lag: 1 }], duration: 3 },
    { id: "E", preds: [{ id: "B", type: "FS", lag: 0 }], duration: 7 },
    { id: "F", preds: [{ id: "B", type: "SS", lag: 2 }], duration: 7 },
    { id: "G", preds: [{ id: "C", type: "FS", lag: 0 }], duration: 7 },
    { id: "H", preds: [{ id: "D", type: "FS", lag: 0 }, { id: "E", type: "SS", lag: 1 }], duration: 7 },
    { id: "I", preds: [{ id: "F", type: "FF", lag: 0 }], duration: 2 },
    { id: "J", preds: [{ id: "G", type: "FS", lag: 0 }, { id: "H", type: "FS", lag: 0 }], duration: 6 },
    { id: "K", preds: [{ id: "I", type: "FS", lag: 0 }, { id: "J", type: "SS", lag: 2 }], duration: 4 },
  ],
  pert_dep: [
    { id: "A", preds: [], o: 2, m: 3, p: 5 },
    { id: "B", preds: [], o: 4, m: 5, p: 7 },
    { id: "C", preds: [{ id: "A", type: "FS", lag: 0 }], o: 3, m: 4, p: 6 },
    { id: "D", preds: [{ id: "A", type: "FS", lag: 1 }], o: 2, m: 3, p: 4 },
    { id: "E", preds: [{ id: "B", type: "FS", lag: 0 }], o: 5, m: 7, p: 10 },
    { id: "F", preds: [{ id: "B", type: "SS", lag: 2 }], o: 5, m: 6, p: 7 },
    { id: "G", preds: [{ id: "C", type: "FS", lag: 0 }], o: 6, m: 7, p: 9 },
    { id: "H", preds: [{ id: "D", type: "FS", lag: 0 }, { id: "E", type: "SS", lag: 1 }], o: 5, m: 7, p: 8 },
    { id: "I", preds: [{ id: "F", type: "FF", lag: 0 }], o: 1, m: 2, p: 3 },
    { id: "J", preds: [{ id: "G", type: "FS", lag: 0 }, { id: "H", type: "FS", lag: 0 }], o: 4, m: 6, p: 9 },
    { id: "K", preds: [{ id: "I", type: "FS", lag: 0 }, { id: "J", type: "SS", lag: 2 }], o: 3, m: 4, p: 5 },
  ],
};

const PANEL_HINTS = {
  cpm:      "CPM · predecessors · duration",
  pert:     "PERT · predecessors · O / M / P",
  cpm_dep:  "CPM · typed dependencies · duration",
  pert_dep: "PERT · typed dependencies · O / M / P",
};

const EXAMPLE_TEXT = {
  cpm: `# Load Dataset
# ID, Dependencies, Duration
dataset = [
    ['A', [], 3],
    ['B', [], 5],
    ['C', ['A'], 4],
    ['D', ['A'], 3],
    ['E', ['B'], 7],
    ['F', ['B'], 7],
    ['G', ['C'], 7],
    ['H', ['D', 'E'], 7],
    ['I', ['F'], 2],
]`,
  pert: `# Load Dataset
# ID, Dependencies, Duration (O), Duration (M), Duration (P)
dataset = [
    ['A', [], 4, 6, 8],
    ['B', ['A'], 3, 4, 6],
    ['C', [], 2, 3, 6],
    ['D', [], 7, 11, 12],
    ['E', ['B', 'C'], 9, 12, 15],
    ['F', ['B', 'C'], 22, 26, 27],
    ['G', ['A'], 6, 7, 8],
    ['H', ['D', 'E', 'G'], 9, 12, 15],
    ['I', ['F', 'H'], 5, 7, 9],
]`,
  cpm_dep: `# Load Dataset
dataset = [
    ('A', [], 3),
    ('B', [], 5),
    ('C', [('A', 'FS', 0)], 4),
    ('D', [('A', 'FS', 1)], 3),
    ('E', [('B', 'FS', 0)], 7),
    ('F', [('B', 'SS', 2)], 7),
    ('G', [('C', 'FS', 0)], 7),
    ('H', [('D', 'FS', 0), ('E', 'SS', 1)], 7),
    ('I', [('F', 'FF', 0)], 2),
    ('J', [('G', 'FS', 0), ('H', 'FS', 0)], 6),
    ('K', [('I', 'FS', 0), ('J', 'SS', 2)], 4),
]`,
  pert_dep: `# Load Dataset
dataset = [
    ('A', [], 2, 3, 5),
    ('B', [], 4, 5, 7),
    ('C', [('A', 'FS', 0)], 3, 4, 6),
    ('D', [('A', 'FS', 1)], 2, 3, 4),
    ('E', [('B', 'FS', 0)], 5, 7, 10),
    ('F', [('B', 'SS', 2)], 5, 6, 7),
    ('G', [('C', 'FS', 0)], 6, 7, 9),
    ('H', [('D', 'FS', 0), ('E', 'SS', 1)], 5, 7, 8),
    ('I', [('F', 'FF', 0)], 1, 2, 3),
    ('J', [('G', 'FS', 0), ('H', 'FS', 0)], 4, 6, 9),
    ('K', [('I', 'FS', 0), ('J', 'SS', 2)], 3, 4, 5),
]`,
};


/* ---------- 3. Method switching + editor ---------- */

function isDep()  { return state.method.endsWith("_dep"); }
function isPert() { return state.method.startsWith("pert"); }

function nextId() {
  // Find first unused single-letter id, then fall back to T-numbered
  const taken = new Set(state.tasks.map(t => t.id));
  for (let c = 65; c <= 90; c++) {
    const id = String.fromCharCode(c);
    if (!taken.has(id)) return id;
  }
  let n = 1;
  while (taken.has("T" + n)) n++;
  return "T" + n;
}

function emptyTaskForMethod() {
  const base = { id: nextId(), preds: [] };
  if (isPert()) Object.assign(base, { o: 1, m: 2, p: 3 });
  else          Object.assign(base, { duration: 1 });
  return base;
}

/** Convert a task between schemas while preserving as much as possible. */
function parseNumericInput(value) {
  if (value === "" || value === null || value === undefined) return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return n;
}

function showBusy(title = "Computing schedule", msg = "Please wait while pyCritical analyzes the project.") {
  if (!refs.busyOverlay) return;
  refs.busyTitle.textContent = title;
  refs.busyMsg.textContent = msg;
  refs.busyOverlay.hidden = false;
  document.body.classList.add("is-busy");
}

function hideBusy() {
  if (!refs.busyOverlay) return;
  refs.busyOverlay.hidden = true;
  document.body.classList.remove("is-busy");
}

async function fetchJsonWithTimeout(url, options = {}, timeoutMs = 20000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const data = await response.json();
    return { response, data };
  } catch (error) {
    if (error && error.name === 'AbortError') {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)} seconds.`);
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

function syncMethodUi() {
  refs.methodTabs.forEach(b => {
    const on = b.dataset.method === state.method;
    b.classList.toggle("is-active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  refs.panelHint.textContent = PANEL_HINTS[state.method];
  refs.probTabBtn.hidden = !isPert();
  if (state.view === "probability" && !isPert()) setView("gantt");
}

function setEditorMode(mode) {
  state.editorMode = mode;
  refs.editorModeTabs.forEach(b => {
    const on = b.dataset.editorMode === mode;
    b.classList.toggle("is-active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  refs.panePaste.hidden = mode !== "paste";
  refs.paneManual.hidden = mode !== "manual";
}

function setPasteStatus(message, tone = "info") {
  if (!refs.pasteStatus) return;
  refs.pasteStatus.textContent = message;
  refs.pasteStatus.classList.remove("is-success", "is-error");
  if (tone === "success") refs.pasteStatus.classList.add("is-success");
  if (tone === "error") refs.pasteStatus.classList.add("is-error");
}

function migrateTask(task, fromMethod, toMethod) {
  const out = { id: task.id };
  // Predecessors: simple list <-> typed list
  const fromDep = fromMethod.endsWith("_dep");
  const toDep   = toMethod.endsWith("_dep");
  if (fromDep && !toDep) {
    out.preds = (task.preds || []).map(p => p.id);
  } else if (!fromDep && toDep) {
    out.preds = (task.preds || []).map(id => ({ id, type: "FS", lag: 0 }));
  } else {
    out.preds = JSON.parse(JSON.stringify(task.preds || []));
  }
  // Duration vs O/M/P
  const fromPert = fromMethod.startsWith("pert");
  const toPert   = toMethod.startsWith("pert");
  if (fromPert && !toPert) {
    // expected duration ~ M
    out.duration = task.m ?? 1;
  } else if (!fromPert && toPert) {
    const d = task.duration ?? 1;
    out.o = Math.max(1, Math.round(d * 0.7 * 10) / 10);
    out.m = d;
    out.p = Math.round(d * 1.4 * 10) / 10;
  } else if (toPert) {
    out.o = task.o; out.m = task.m; out.p = task.p;
  } else {
    out.duration = task.duration;
  }
  return out;
}

function setMethod(method) {
  if (state.method === method) return;
  const old = state.method;
  state.method = method;
  state.tasks = state.tasks.map(t => migrateTask(t, old, method));
  syncMethodUi();
  renderEditor();
  state.result = null;
  showEmpty();
  updateKpiStrip(null);
}

function renderEditorHead() {
  const cols = [];
  cols.push(`<th class="col-id">ID</th>`);
  cols.push(`<th>Predecessors</th>`);
  if (isPert()) {
    cols.push(`<th class="col-num">O</th>`);
    cols.push(`<th class="col-num">M</th>`);
    cols.push(`<th class="col-num">P</th>`);
  } else {
    cols.push(`<th class="col-num">Duration</th>`);
  }
  cols.push(`<th class="col-actions"></th>`);
  refs.taskThead.innerHTML = `<tr>${cols.join("")}</tr>`;
}

function renderEditorBody() {
  refs.taskTbody.innerHTML = "";
  state.tasks.forEach((task, idx) => {
    const tr = document.createElement("tr");
    tr.dataset.idx = String(idx);

    // ID
    const tdId = document.createElement("td");
    tdId.className = "col-id";
    const idIn = document.createElement("input");
    idIn.type = "text";
    idIn.value = task.id;
    idIn.placeholder = "ID";
    idIn.addEventListener("input", e => { task.id = e.target.value; });
    tdId.appendChild(idIn);
    tr.appendChild(tdId);

    // Predecessors
    const tdPreds = document.createElement("td");
    if (isDep()) {
      tdPreds.className = "preds-cell";
      tdPreds.appendChild(buildDepPredsEditor(task));
    } else {
      const inp = document.createElement("input");
      inp.type = "text";
      inp.placeholder = "—";
      inp.value = (task.preds || []).join(", ");
      inp.addEventListener("input", e => {
        task.preds = e.target.value
          .split(/[, ]+/)
          .map(s => s.trim())
          .filter(Boolean);
      });
      tdPreds.appendChild(inp);
    }
    tr.appendChild(tdPreds);

    // Numeric fields
    if (isPert()) {
      ["o", "m", "p"].forEach(key => {
        const td = document.createElement("td");
        td.className = "col-num";
        const inp = document.createElement("input");
        inp.type = "number"; inp.step = "any"; inp.min = "0"; inp.inputMode = "decimal";
        inp.value = task[key] ?? "";
        inp.addEventListener("input", e => { task[key] = parseNumericInput(e.target.value); });
        td.appendChild(inp);
        tr.appendChild(td);
      });
    } else {
      const td = document.createElement("td");
      td.className = "col-num";
      const inp = document.createElement("input");
      inp.type = "number"; inp.step = "any"; inp.min = "0"; inp.inputMode = "decimal";
      inp.value = task.duration ?? "";
      inp.addEventListener("input", e => { task.duration = parseNumericInput(e.target.value); });
      td.appendChild(inp);
      tr.appendChild(td);
    }

    // Delete
    const tdAct = document.createElement("td");
    tdAct.className = "col-actions";
    const del = document.createElement("button");
    del.className = "row-del";
    del.title = "Remove";
    del.innerHTML = "×";
    del.addEventListener("click", () => {
      state.tasks.splice(idx, 1);
      renderEditorBody();
    });
    tdAct.appendChild(del);
    tr.appendChild(tdAct);

    refs.taskTbody.appendChild(tr);
  });
}

function buildDepPredsEditor(task) {
  const stack = document.createElement("div");
  stack.className = "preds-stack";
  (task.preds || []).forEach((p, i) => stack.appendChild(buildPredChip(task, i)));
  const add = document.createElement("button");
  add.className = "pred-add";
  add.textContent = "+ dep";
  add.addEventListener("click", () => {
    task.preds.push({ id: "", type: "FS", lag: 0 });
    renderEditorBody();
  });
  stack.appendChild(add);
  return stack;
}

function buildPredChip(task, i) {
  const p = task.preds[i];
  const chip = document.createElement("span");
  chip.className = "pred-chip";

  const idIn = document.createElement("input");
  idIn.className = "pred-chip__id";
  idIn.type = "text";
  idIn.value = p.id;
  idIn.placeholder = "id";
  idIn.addEventListener("input", e => { p.id = e.target.value.trim(); });

  const typeSel = document.createElement("select");
  ["FS", "SS", "FF", "SF"].forEach(t => {
    const opt = document.createElement("option");
    opt.value = t; opt.textContent = t;
    if (t === p.type) opt.selected = true;
    typeSel.appendChild(opt);
  });
  typeSel.addEventListener("change", e => { p.type = e.target.value; });

  const lagIn = document.createElement("input");
  lagIn.className = "lag";
  lagIn.type = "number"; lagIn.step = "any"; lagIn.inputMode = "decimal";
  lagIn.value = p.lag ?? 0;
  lagIn.title = "Lag (positive or negative integer)";
  lagIn.addEventListener("input", e => { p.lag = parseNumericInput(e.target.value) ?? 0; });

  const x = document.createElement("button");
  x.className = "pred-chip__remove";
  x.innerHTML = "×";
  x.addEventListener("click", () => {
    task.preds.splice(i, 1);
    renderEditorBody();
  });

  chip.append(idIn, typeSel, lagIn, x);
  return chip;
}

function renderEditor() {
  renderEditorHead();
  renderEditorBody();
}

async function parseDatasetFromPaste({ switchToManual = true, showOverlay = true } = {}) {
  const raw = refs.datasetPaste.value.trim();
  if (!raw) {
    setPasteStatus("Paste a dataset first.", "error");
    setEditorMode("paste");
    return false;
  }

  let resp, json;
  try {
    if (showOverlay) {
      showBusy("Parsing dataset", "Please wait while pyCritical detects the schema and converts the pasted data.");
    }
    ({ response: resp, data: json } = await fetchJsonWithTimeout("/api/parse_dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: raw }),
    }, 15000));
  } catch (e) {
    if (showOverlay) hideBusy();
    setPasteStatus("Parsing failed: " + e.message, "error");
    showError("Parse error: " + e.message);
    setEditorMode("paste");
    return false;
  } finally {
    if (showOverlay) hideBusy();
  }

  if (!resp?.ok || !json.ok) {
    const msg = json?.error || "Unknown parse error.";
    setPasteStatus(msg, "error");
    showError(msg);
    setEditorMode("paste");
    return false;
  }

  state.method = json.method;
  state.tasks = JSON.parse(JSON.stringify(json.tasks || []));
  state.result = null;
  state.pasteDirty = false;
  refs.datasetPaste.value = json.normalized_text || raw;
  syncMethodUi();
  renderEditor();
  showEmpty();
  updateKpiStrip(null);
  hideError();
  setPasteStatus(`Parsed ${json.count} task${json.count === 1 ? "" : "s"} as ${json.method.toUpperCase()} from ${json.source} input.`, "success");
  if (switchToManual) setEditorMode("manual");
  return true;
}


/* ---------- 4. Compute ---------- */

async function compute() {
  if (state.editorMode === "paste" && refs.datasetPaste.value.trim() && (state.pasteDirty || state.tasks.length === 0)) {
    const ok = await parseDatasetFromPaste({ switchToManual: true });
    if (!ok) return;
  }

  const ids = new Set();
  for (const t of state.tasks) {
    const tid = String(t.id || "").trim();
    if (!tid) {
      return showError("Every task needs a non-empty id.");
    }
    if (ids.has(tid)) return showError(`Duplicate task id: ${tid}`);
    ids.add(tid);

    if (isPert()) {
      for (const key of ["o", "m", "p"]) {
        if (!Number.isFinite(t[key]) || t[key] <= 0) {
          return showError(`Task ${tid}: ${key.toUpperCase()} must be a positive number.`);
        }
      }
    } else if (!Number.isFinite(t.duration) || t.duration <= 0) {
      return showError(`Task ${tid}: Duration must be a positive number.`);
    }

    if (isDep()) {
      for (const dep of (t.preds || [])) {
        const pid = String(dep.id || "").trim();
        if (!pid) {
          return showError(`Task ${tid}: each dependency needs a predecessor id.`);
        }
        if (!Number.isFinite(dep.lag ?? 0)) {
          return showError(`Task ${tid}: lag values must be numeric.`);
        }
      }
    }
  }

  if (state.tasks.length === 0) return showError("Add at least one task.");

  const endpoint = "/api/" + state.method;
  let resp, json;
  const runLabel = refs.btnRun.querySelector("span");
  try {
    refs.btnRun.disabled = true;
    if (runLabel) runLabel.textContent = "Computing…";
    showBusy("Computing schedule", "Please wait while pyCritical analyzes the project.");
    await new Promise(resolve => setTimeout(resolve, 120));
    ({ response: resp, data: json } = await fetchJsonWithTimeout(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tasks: state.tasks }),
    }, 30000));
  } catch (e) {
    return showError("Network error: " + e.message);
  } finally {
    hideBusy();
    refs.btnRun.disabled = false;
    if (runLabel) runLabel.textContent = "Compute ⟶";
  }

  if (!resp?.ok || !json.ok) return showError(json?.error || "Unknown error.");

  state.result = json;
  hideError();
  hideEmpty();
  updateKpiStrip(json);
  setView(state.view);
}


/* ---------- 5. KPI + tab visibility ---------- */

function fmt(n, d = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  // Show whole number if essentially an integer
  if (Math.abs(n - Math.round(n)) < 1e-9) return String(Math.round(n));
  return n.toFixed(d).replace(/\.?0+$/, "");
}

function updateKpiStrip(res) {
  if (!res) {
    refs.kpiDuration.textContent = "—";
    refs.kpiTasks.textContent    = "—";
    refs.kpiPath.textContent     = "—";
    refs.kpiStdDevWrap.hidden    = true;
    return;
  }
  refs.kpiDuration.textContent = fmt(res.finish_time);
  refs.kpiTasks.textContent    = String(res.tasks.length);
  const path = res.tasks.filter(t => t.critical).map(t => t.id);
  refs.kpiPath.textContent     = path.length ? path.join(" → ") : "—";

  if (res.std_dev !== undefined) {
    refs.kpiStdDev.textContent = fmt(res.std_dev, 3);
    refs.kpiStdDevWrap.hidden  = false;
  } else {
    refs.kpiStdDevWrap.hidden  = true;
  }
}

function setView(view) {
  state.view = view;
  refs.vizTabs.forEach(t => {
    const on = t.dataset.view === view;
    t.classList.toggle("is-active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  [refs.paneGantt, refs.paneNetwork, refs.paneProb, refs.paneTable].forEach(p => {
    p.hidden = (p.dataset.view !== view);
  });
  if (state.result) renderActiveView();
}

function renderActiveView() {
  if (!state.result) return;
  hideEmpty();
  hideError();
  [refs.paneGantt, refs.paneNetwork, refs.paneProb, refs.paneTable].forEach(p => {
    p.hidden = (p.dataset.view !== state.view);
  });
  switch (state.view) {
    case "gantt":
      renderGantt(state.result);
      break;
    case "network":
      renderNetwork(state.result);
      break;
    case "probability":
      renderProbability(state.result);
      break;
    case "table":
      renderTable(state.result);
      break;
  }
}

function showEmpty() {
  refs.vizEmpty.hidden = false;
  [refs.paneGantt, refs.paneNetwork, refs.paneProb, refs.paneTable].forEach(p => p.hidden = true);
}
function hideEmpty() { refs.vizEmpty.hidden = true; }
function showError(msg) {
  refs.vizError.hidden = false;
  refs.vizErrorMsg.textContent = msg;
  [refs.paneGantt, refs.paneNetwork, refs.paneProb, refs.paneTable, refs.vizEmpty].forEach(p => p.hidden = true);
}
function hideError() { refs.vizError.hidden = true; }


/* ---------- 6. Gantt visualization ---------- */
/* Built fresh as SVG: bars, slack tails, dependency arrows, hover tooltips. */

function renderGantt(res) {
  const wrap = refs.ganttWrap;
  wrap.innerHTML = "";

  const tasks   = res.tasks.slice();   // schedule rows in their natural order
  const dataset = res.dataset || [];

  const showSlack = refs.ganttShowSlack.checked;
  const showDeps  = refs.ganttShowDeps.checked;

  // Lookup: id -> task index in `tasks`, and id -> dataset entry
  const idxOf = new Map(tasks.map((t, i) => [t.id, i]));
  const dsOf  = new Map(dataset.map(d => [d.id, d]));

  // Geometry
  const finish = Math.max(1, res.finish_time);
  const padding = 0.04 * finish;
  const tMax = finish + padding;

  const marginL = 84;
  const marginR = 24;
  const marginT = 36;
  const marginB = 32;
  const rowH    = 34;
  const barH    = 22;
  const innerH  = tasks.length * rowH;
  const innerW  = Math.max(640, wrap.clientWidth - marginL - marginR - 16);
  const W = innerW + marginL + marginR;
  const H = innerH + marginT + marginB;

  const xScale = (t) => marginL + (t / tMax) * innerW;
  const yOf    = (i) => marginT + i * rowH + rowH / 2;

  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("class", "gantt-svg");
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  // Defs (gradients + arrowheads)
  const defs = document.createElementNS(SVGNS, "defs");
  defs.innerHTML = `
    <linearGradient id="barNormGrad" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%"  stop-color="#94e0f3"/>
      <stop offset="100%" stop-color="#3a8ba3"/>
    </linearGradient>
    <linearGradient id="barCritGrad" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%"  stop-color="#ffd089"/>
      <stop offset="100%" stop-color="#c9802a"/>
    </linearGradient>
    <marker id="arrowDef" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 Z" fill="rgba(245,224,168,0.55)"/>
    </marker>
    <marker id="arrowCrit" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 Z" fill="#f5b042"/>
    </marker>
  `;
  svg.appendChild(defs);

  // Time grid + axis
  const ticks = niceTicks(0, tMax, 8);
  const grid = document.createElementNS(SVGNS, "g");
  ticks.forEach((t, i) => {
    const x = xScale(t);
    const ln = document.createElementNS(SVGNS, "line");
    ln.setAttribute("class", i === 0 || i === ticks.length - 1 ? "grid-line grid-line--major" : "grid-line");
    ln.setAttribute("x1", x); ln.setAttribute("x2", x);
    ln.setAttribute("y1", marginT - 8); ln.setAttribute("y2", H - marginB);
    grid.appendChild(ln);

    const lbl = document.createElementNS(SVGNS, "text");
    lbl.setAttribute("class", "axis-label");
    lbl.setAttribute("x", x); lbl.setAttribute("y", H - marginB + 18);
    lbl.setAttribute("text-anchor", "middle");
    lbl.textContent = fmt(t);
    grid.appendChild(lbl);
  });
  svg.appendChild(grid);

  // Row backgrounds (for hover hit area)
  tasks.forEach((t, i) => {
    const rect = document.createElementNS(SVGNS, "rect");
    rect.setAttribute("class", "row-bg");
    rect.setAttribute("x", marginL);
    rect.setAttribute("y", marginT + i * rowH);
    rect.setAttribute("width", innerW);
    rect.setAttribute("height", rowH);
    svg.appendChild(rect);
  });

  // Task labels (left)
  tasks.forEach((t, i) => {
    const txt = document.createElementNS(SVGNS, "text");
    txt.setAttribute("class", "row-label" + (t.critical ? " is-critical" : ""));
    txt.setAttribute("x", marginL - 12);
    txt.setAttribute("y", yOf(i));
    txt.setAttribute("text-anchor", "end");
    txt.setAttribute("dominant-baseline", "middle");
    txt.textContent = t.id;
    svg.appendChild(txt);
  });

  // Slack tails
  if (showSlack) {
    tasks.forEach((t, i) => {
      if (!(t.Slack > 0) || t.LF === undefined || t.LF === null) return;
      const x1 = xScale(t.EF);
      const x2 = xScale(t.LF);
      const y  = yOf(i);
      const ln = document.createElementNS(SVGNS, "line");
      ln.setAttribute("class", "slack-line");
      ln.setAttribute("x1", x1); ln.setAttribute("x2", x2);
      ln.setAttribute("y1", y);  ln.setAttribute("y2", y);
      svg.appendChild(ln);
      const cap = document.createElementNS(SVGNS, "circle");
      cap.setAttribute("class", "slack-cap");
      cap.setAttribute("cx", x2); cap.setAttribute("cy", y);
      cap.setAttribute("r", 2.5);
      svg.appendChild(cap);
    });
  }

  // Dependency arrows
  if (showDeps) {
    tasks.forEach((t, i) => {
      const ds = dsOf.get(t.id);
      if (!ds) return;
      const preds = isDep() ? ds.preds : (ds.preds || []).map(id => ({ id, type: "FS", lag: 0 }));
      preds.forEach(p => {
        const j = idxOf.get(p.id);
        if (j === undefined) return;
        const pred = tasks[j];

        const isCritEdge = t.critical && pred.critical;
        const dt = p.type || "FS";

        // Anchor points based on dep type
        const fromX = (dt === "SS" || dt === "SF") ? xScale(pred.ES) : xScale(pred.EF);
        const fromY = yOf(j) + (i > j ? 8 : -8);
        const toX   = (dt === "SS" || dt === "FS") ? xScale(t.ES) : xScale(t.EF);
        const toY   = yOf(i) + (i > j ? -8 : 8);

        // Right-angle path: from→ midX → toX
        const midX = (fromX + toX) / 2;
        const path = document.createElementNS(SVGNS, "path");
        path.setAttribute("class", "dep-arrow" + (isCritEdge ? " dep-arrow--crit" : ""));
        path.setAttribute("d", `M ${fromX} ${fromY} L ${midX} ${fromY} L ${midX} ${toY} L ${toX} ${toY}`);
        path.setAttribute("marker-end", `url(#${isCritEdge ? "arrowCrit" : "arrowDef"})`);
        svg.appendChild(path);

        if (isDep() && (dt !== "FS" || (p.lag || 0) !== 0)) {
          const lblX = midX;
          const lblY = (fromY + toY) / 2;
          const lbl = document.createElementNS(SVGNS, "text");
          lbl.setAttribute("class", "dep-label");
          lbl.setAttribute("x", lblX + 4);
          lbl.setAttribute("y", lblY);
          lbl.setAttribute("dominant-baseline", "middle");
          lbl.textContent = `${dt}${p.lag >= 0 ? "+" : ""}${p.lag}`;
          svg.appendChild(lbl);
        }
      });
    });
  }

  // Bars (drawn last, on top)
  tasks.forEach((t, i) => {
    const x  = xScale(t.ES);
    const w  = Math.max(2, xScale(t.EF) - xScale(t.ES));
    const y  = yOf(i) - barH / 2;
    const g  = document.createElementNS(SVGNS, "g");
    g.style.cursor = "pointer";

    const r  = document.createElementNS(SVGNS, "rect");
    r.setAttribute("class", "bar " + (t.critical ? "bar--crit" : "bar--norm"));
    r.setAttribute("x", x); r.setAttribute("y", y);
    r.setAttribute("width", w); r.setAttribute("height", barH);
    r.setAttribute("rx", 4);
    g.appendChild(r);

    const dur = (t.EF - t.ES);
    if (w > 28) {
      const lbl = document.createElementNS(SVGNS, "text");
      const inverse = w < 60;
      lbl.setAttribute("class", "bar-label" + (t.critical ? " bar-label--crit" : (inverse ? " bar-label--inverse" : "")));
      lbl.setAttribute("x", x + w / 2);
      lbl.setAttribute("y", y + barH / 2);
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("dominant-baseline", "middle");
      lbl.textContent = fmt(dur);
      g.appendChild(lbl);
    }

    g.addEventListener("mousemove", e => showTooltipForTask(e, t));
    g.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(g);
  });

  wrap.appendChild(svg);
}

function niceTicks(min, max, target) {
  const range = max - min;
  if (range <= 0) return [min];
  const rough = range / target;
  const pow   = Math.pow(10, Math.floor(Math.log10(rough)));
  const m     = rough / pow;
  let step;
  if (m < 1.5) step = 1 * pow;
  else if (m < 3) step = 2 * pow;
  else if (m < 7) step = 5 * pow;
  else step = 10 * pow;
  const out = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + 1e-9; v += step) {
    out.push(Math.round(v / step) * step);
  }
  if (out[0] > min) out.unshift(min);
  if (out[out.length - 1] < max) out.push(max);
  return out;
}


/* ---------- 7. Network visualization (Activity-on-Node) ---------- */
/* Layered topological layout, then draggable nodes. */

const NET = {
  nodeW: 130,
  nodeH: 96,
  hSpacing: 60,    // gap between layers
  vSpacing: 24,    // gap between nodes in a layer
};

function computeLayers(tasks, dataset) {
  // Each task's "rank" = longest predecessor chain length.
  const idxOf = new Map(tasks.map((t, i) => [t.id, i]));
  const preds = new Map();
  dataset.forEach(d => {
    const ps = (d.preds || []).map(p => (typeof p === "string" ? p : p.id));
    preds.set(d.id, ps);
  });
  const rank = new Map();
  function rankOf(id, stack = new Set()) {
    if (rank.has(id)) return rank.get(id);
    if (stack.has(id)) return 0;            // cycle safety
    stack.add(id);
    const ps = preds.get(id) || [];
    const r = ps.length === 0 ? 0 : 1 + Math.max(...ps.map(p => rankOf(p, stack)));
    rank.set(id, r);
    stack.delete(id);
    return r;
  }
  tasks.forEach(t => rankOf(t.id));

  const layers = [];
  rank.forEach((r, id) => {
    if (!layers[r]) layers[r] = [];
    layers[r].push(id);
  });
  // Sort each layer by ES then by id for stability
  layers.forEach(layer => {
    layer.sort((a, b) => {
      const ta = tasks[idxOf.get(a)], tb = tasks[idxOf.get(b)];
      if (ta.ES !== tb.ES) return ta.ES - tb.ES;
      return a.localeCompare(b);
    });
  });
  return layers;
}

function renderNetwork(res) {
  const wrap = refs.networkWrap;
  wrap.innerHTML = "";

  const tasks   = res.tasks.slice();
  const dataset = res.dataset || [];
  const idxOf   = new Map(tasks.map((t, i) => [t.id, i]));
  const dsOf    = new Map(dataset.map(d => [d.id, d]));

  const layers = computeLayers(tasks, dataset);

  // Compute initial positions
  const positions = new Map();
  let maxRows = Math.max(...layers.map(l => l.length));
  const colW = NET.nodeW + NET.hSpacing;
  const rowH = NET.nodeH + NET.vSpacing;

  layers.forEach((layer, ci) => {
    const totalH = layer.length * rowH;
    const offsetY = (maxRows * rowH - totalH) / 2;
    layer.forEach((id, ri) => {
      positions.set(id, {
        x: 60 + ci * colW,
        y: 40 + offsetY + ri * rowH,
      });
    });
  });

  const totalW = 120 + layers.length * colW;
  const totalH = 80 + maxRows * rowH;

  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("class", "network-svg");
  svg.setAttribute("viewBox", `0 0 ${totalW} ${totalH}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const defs = document.createElementNS(SVGNS, "defs");
  defs.innerHTML = `
    <marker id="netArrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 Z" fill="rgba(245,224,168,0.55)"/>
    </marker>
    <marker id="netArrowCrit" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 Z" fill="#f5b042"/>
    </marker>
  `;
  svg.appendChild(defs);

  // Edge layer + Node layer (so nodes always sit on top)
  const gEdges = document.createElementNS(SVGNS, "g");
  const gNodes = document.createElementNS(SVGNS, "g");
  svg.appendChild(gEdges);
  svg.appendChild(gNodes);

  // Build edges first
  const edges = [];
  tasks.forEach(t => {
    const ds = dsOf.get(t.id);
    if (!ds) return;
    const preds = isDep() ? ds.preds : (ds.preds || []).map(id => ({ id, type: "FS", lag: 0 }));
    preds.forEach(p => {
      const ji = idxOf.get(p.id);
      if (ji === undefined) return;
      const pred = tasks[ji];
      const crit = t.critical && pred.critical;
      edges.push({ from: p.id, to: t.id, type: p.type, lag: p.lag || 0, crit });
    });
  });

  function edgeEndpoints(from, to) {
    const a = positions.get(from), b = positions.get(to);
    return {
      x1: a.x + NET.nodeW,           // right edge of source
      y1: a.y + NET.nodeH / 2,
      x2: b.x,                       // left edge of target
      y2: b.y + NET.nodeH / 2,
    };
  }

  function pathFor(e) {
    const { x1, y1, x2, y2 } = edgeEndpoints(e.from, e.to);
    const dx = Math.max(20, (x2 - x1) * 0.5);
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  }

  function midpoint(e) {
    const { x1, y1, x2, y2 } = edgeEndpoints(e.from, e.to);
    return { x: (x1 + x2) / 2, y: (y1 + y2) / 2 };
  }

  // Render edges
  const edgeEls = edges.map(e => {
    const path = document.createElementNS(SVGNS, "path");
    path.setAttribute("class", "net-edge" + (e.crit ? " net-edge--crit" : ""));
    path.setAttribute("d", pathFor(e));
    path.setAttribute("marker-end", `url(#${e.crit ? "netArrowCrit" : "netArrow"})`);
    gEdges.appendChild(path);

    let lblEl = null;
    if (isDep() && (e.type !== "FS" || e.lag !== 0)) {
      const m = midpoint(e);
      const lbl = document.createElementNS(SVGNS, "text");
      lbl.setAttribute("class", "net-edge-label");
      lbl.setAttribute("x", m.x); lbl.setAttribute("y", m.y - 4);
      lbl.setAttribute("text-anchor", "middle");
      lbl.textContent = `${e.type}${e.lag >= 0 ? "+" : ""}${e.lag}`;
      gEdges.appendChild(lbl);
      lblEl = lbl;
    }
    return { e, path, lblEl };
  });

  // Render nodes
  const nodeEls = new Map();
  tasks.forEach(t => {
    const pos = positions.get(t.id);
    const g = document.createElementNS(SVGNS, "g");
    g.setAttribute("class", "net-node" + (t.critical ? " net-node--crit" : ""));
    g.setAttribute("transform", `translate(${pos.x}, ${pos.y})`);
    g.dataset.id = t.id;

    const card = document.createElementNS(SVGNS, "rect");
    card.setAttribute("class", "net-node-card");
    card.setAttribute("x", 0); card.setAttribute("y", 0);
    card.setAttribute("width", NET.nodeW); card.setAttribute("height", NET.nodeH);
    card.setAttribute("rx", 8);
    g.appendChild(card);

    // ID + critical sigil
    const id = document.createElementNS(SVGNS, "text");
    id.setAttribute("class", "net-node-id");
    id.setAttribute("x", 12); id.setAttribute("y", 26);
    id.textContent = t.id;
    g.appendChild(id);

    if (t.critical) {
      const star = document.createElementNS(SVGNS, "text");
      star.setAttribute("class", "net-node-id");
      star.setAttribute("x", NET.nodeW - 12); star.setAttribute("y", 22);
      star.setAttribute("text-anchor", "end");
      star.setAttribute("font-size", "12");
      star.textContent = "◆";
      g.appendChild(star);
    }

    // Divider
    const div1 = document.createElementNS(SVGNS, "line");
    div1.setAttribute("class", "net-node-divider");
    div1.setAttribute("x1", 0); div1.setAttribute("x2", NET.nodeW);
    div1.setAttribute("y1", 36); div1.setAttribute("y2", 36);
    g.appendChild(div1);

    // ES | EF row
    addLabel(g, "ES", 12, 50);
    addValue(g, fmt(t.ES), 38, 50);
    addLabel(g, "EF", 76, 50);
    addValue(g, fmt(t.EF), 100, 50);

    // LS | LF row
    addLabel(g, "LS", 12, 66);
    addValue(g, fmt(t.LS), 38, 66);
    addLabel(g, "LF", 76, 66);
    addValue(g, fmt(t.LF), 100, 66);

    const div2 = document.createElementNS(SVGNS, "line");
    div2.setAttribute("class", "net-node-divider");
    div2.setAttribute("x1", 0); div2.setAttribute("x2", NET.nodeW);
    div2.setAttribute("y1", 74); div2.setAttribute("y2", 74);
    g.appendChild(div2);

    // Slack
    addLabel(g, "Slack", 12, 88);
    addValue(g, fmt(t.Slack), 50, 88);

    g.addEventListener("mousemove", e => showTooltipForTask(e, t));
    g.addEventListener("mouseleave", hideTooltip);
    gNodes.appendChild(g);
    nodeEls.set(t.id, g);
  });

  function addLabel(g, txt, x, y) {
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("class", "net-node-meta");
    t.setAttribute("x", x); t.setAttribute("y", y);
    t.textContent = txt;
    g.appendChild(t);
  }
  function addValue(g, txt, x, y) {
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("class", "net-node-val");
    t.setAttribute("x", x); t.setAttribute("y", y);
    t.textContent = txt;
    g.appendChild(t);
  }

  wrap.appendChild(svg);

  // ---- Drag interaction (per-node) ----
  let drag = null;
  function svgPoint(evt) {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }
  nodeEls.forEach((el, id) => {
    el.addEventListener("mousedown", evt => {
      evt.preventDefault();
      const start = svgPoint(evt);
      const orig = positions.get(id);
      drag = { id, dx: start.x - orig.x, dy: start.y - orig.y };
    });
  });
  svg.addEventListener("mousemove", evt => {
    if (!drag) return;
    const p = svgPoint(evt);
    const np = { x: p.x - drag.dx, y: p.y - drag.dy };
    positions.set(drag.id, np);
    nodeEls.get(drag.id).setAttribute("transform", `translate(${np.x}, ${np.y})`);
    // Update edges touching this node
    edgeEls.forEach(({ e, path, lblEl }) => {
      if (e.from === drag.id || e.to === drag.id) {
        path.setAttribute("d", pathFor(e));
        if (lblEl) {
          const m = midpoint(e);
          lblEl.setAttribute("x", m.x);
          lblEl.setAttribute("y", m.y - 4);
        }
      }
    });
    hideTooltip();
  });
  window.addEventListener("mouseup", () => { drag = null; });
}


/* ---------- 8. Probability visualization (PERT only) ---------- */
/* Bell curve with two-way scrubbing between date and probability. */

const probState = {
  mean: 0, sigma: 1, currentDate: 0,
};

// Standard normal PDF & CDF (Abramowitz–Stegun approximation for CDF)
function normPdf(x, mu, sigma) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}
function normCdf(x, mu, sigma) {
  return 0.5 * (1 + erf((x - mu) / (sigma * Math.SQRT2)));
}
function normInvCdf(p, mu, sigma) {
  // Beasley-Springer-Moro
  if (p <= 0) return -Infinity;
  if (p >= 1) return  Infinity;
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969,
              138.357751867269, -30.6647980661472, 2.50662827745924];
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887,
              66.8013118877197, -13.2806815528857];
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
             -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143,
             3.75440866190742];
  let q, r, x;
  if (p < 0.02425) {
    q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
        ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
  } else if (p < 0.97575) {
    q = p - 0.5;
    r = q * q;
    x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q /
        (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
         ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
  }
  return mu + sigma * x;
}

function renderProbability(res) {
  const wrap = refs.probChartWrap;
  wrap.innerHTML = "";

  const mean = res.finish_time;
  const sigma = res.std_dev || 0;
  probState.mean = mean;
  probState.sigma = sigma;

  refs.probMean.textContent = fmt(mean, 3);
  refs.probSigma.textContent = fmt(sigma, 3);

  if (sigma <= 0) {
    wrap.innerHTML = `<div style="display:grid;place-items:center;height:100%;color:var(--ink-3);font-family:var(--font-mono);font-size:12px;text-align:center;padding:32px;">
      Standard deviation is zero — probability is degenerate.<br/>
      Add tasks with O ≠ P along the critical path.</div>`;
    return;
  }

  if (probState.currentDate <= 0 || Math.abs(probState.currentDate - mean) > 5 * sigma) {
    probState.currentDate = mean;
  }

  const W = wrap.clientWidth, H = wrap.clientHeight;
  const marginL = 56, marginR = 24, marginT = 24, marginB = 40;
  const innerW = W - marginL - marginR;
  const innerH = H - marginT - marginB;

  const xMin = mean - 4 * sigma;
  const xMax = mean + 4 * sigma;
  const xScale = x => marginL + (x - xMin) / (xMax - xMin) * innerW;
  const xInv   = px => xMin + (px - marginL) / innerW * (xMax - xMin);

  // Curve sampling
  const N = 240;
  const samples = [];
  let yMax = 0;
  for (let i = 0; i <= N; i++) {
    const x = xMin + (i / N) * (xMax - xMin);
    const y = normPdf(x, mean, sigma);
    samples.push({ x, y });
    if (y > yMax) yMax = y;
  }
  const yScale = y => marginT + innerH - (y / yMax) * innerH * 0.92;

  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("class", "prob-svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const defs = document.createElementNS(SVGNS, "defs");
  defs.innerHTML = `
    <linearGradient id="probFillGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="rgba(95,191,217,0.32)"/>
      <stop offset="100%" stop-color="rgba(95,191,217,0.04)"/>
    </linearGradient>
    <linearGradient id="probFillAccent" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="rgba(245,176,66,0.50)"/>
      <stop offset="100%" stop-color="rgba(245,176,66,0.05)"/>
    </linearGradient>
  `;
  svg.appendChild(defs);

  // Cumulative shaded area (left of marker)
  const accentArea = document.createElementNS(SVGNS, "path");
  accentArea.setAttribute("class", "curve-fill curve-fill--accent");
  svg.appendChild(accentArea);

  // Full area under curve (faint cyan)
  const baseArea = document.createElementNS(SVGNS, "path");
  baseArea.setAttribute("class", "curve-fill");
  let baseD = `M ${xScale(xMin)} ${yScale(0)} `;
  samples.forEach(s => baseD += `L ${xScale(s.x)} ${yScale(s.y)} `);
  baseD += `L ${xScale(xMax)} ${yScale(0)} Z`;
  baseArea.setAttribute("d", baseD);
  svg.appendChild(baseArea);

  // Curve line
  const curve = document.createElementNS(SVGNS, "path");
  curve.setAttribute("class", "curve");
  let curveD = "";
  samples.forEach((s, i) => curveD += `${i === 0 ? "M" : "L"} ${xScale(s.x)} ${yScale(s.y)} `);
  curve.setAttribute("d", curveD);
  svg.appendChild(curve);

  // X axis
  const axis = document.createElementNS(SVGNS, "line");
  axis.setAttribute("class", "axis");
  axis.setAttribute("x1", marginL); axis.setAttribute("x2", W - marginR);
  axis.setAttribute("y1", yScale(0)); axis.setAttribute("y2", yScale(0));
  svg.appendChild(axis);

  // Sigma ticks
  for (let k = -3; k <= 3; k++) {
    const tx = mean + k * sigma;
    if (tx < xMin || tx > xMax) continue;
    const tk = document.createElementNS(SVGNS, "line");
    tk.setAttribute("class", "axis");
    tk.setAttribute("x1", xScale(tx)); tk.setAttribute("x2", xScale(tx));
    tk.setAttribute("y1", yScale(0)); tk.setAttribute("y2", yScale(0) + 5);
    svg.appendChild(tk);
    const lbl = document.createElementNS(SVGNS, "text");
    lbl.setAttribute("class", "axis-tick");
    lbl.setAttribute("x", xScale(tx)); lbl.setAttribute("y", yScale(0) + 18);
    lbl.setAttribute("text-anchor", "middle");
    lbl.textContent = fmt(tx, 2);
    svg.appendChild(lbl);
  }

  // Mean line
  const meanLn = document.createElementNS(SVGNS, "line");
  meanLn.setAttribute("class", "mean-line");
  meanLn.setAttribute("x1", xScale(mean)); meanLn.setAttribute("x2", xScale(mean));
  meanLn.setAttribute("y1", marginT); meanLn.setAttribute("y2", yScale(0));
  svg.appendChild(meanLn);
  const meanLbl = document.createElementNS(SVGNS, "text");
  meanLbl.setAttribute("class", "mean-label");
  meanLbl.setAttribute("x", xScale(mean) + 6); meanLbl.setAttribute("y", marginT + 12);
  meanLbl.textContent = `μ = ${fmt(mean)}`;
  svg.appendChild(meanLbl);

  // Marker
  const markerLn = document.createElementNS(SVGNS, "line");
  markerLn.setAttribute("class", "marker-line");
  svg.appendChild(markerLn);
  const markerHandle = document.createElementNS(SVGNS, "circle");
  markerHandle.setAttribute("class", "marker-handle");
  markerHandle.setAttribute("r", 7);
  svg.appendChild(markerHandle);
  const markerProbLbl = document.createElementNS(SVGNS, "text");
  markerProbLbl.setAttribute("class", "mean-label");
  markerProbLbl.setAttribute("fill", "var(--amber-hi)");
  markerProbLbl.style.fill = "#ffd089";
  svg.appendChild(markerProbLbl);

  function updateMarker(date) {
    date = Math.max(xMin, Math.min(xMax, date));
    probState.currentDate = date;
    const px = xScale(date);

    // Shaded area to the left of marker
    let d = `M ${xScale(xMin)} ${yScale(0)} `;
    for (let i = 0; i <= N; i++) {
      const sx = xMin + (i / N) * (xMax - xMin);
      if (sx > date) break;
      d += `L ${xScale(sx)} ${yScale(normPdf(sx, mean, sigma))} `;
    }
    d += `L ${px} ${yScale(normPdf(date, mean, sigma))} L ${px} ${yScale(0)} Z`;
    accentArea.setAttribute("d", d);

    markerLn.setAttribute("x1", px); markerLn.setAttribute("x2", px);
    markerLn.setAttribute("y1", marginT); markerLn.setAttribute("y2", yScale(0));
    markerHandle.setAttribute("cx", px);
    markerHandle.setAttribute("cy", yScale(normPdf(date, mean, sigma)));

    const p = normCdf(date, mean, sigma);
    const z = (date - mean) / sigma;
    refs.probDate.value  = fmt(date, 4);
    refs.probValue.value = fmt(p, 4);
    refs.probZ.textContent = fmt(z, 3);

    markerProbLbl.setAttribute("x", Math.min(px + 8, W - marginR - 60));
    markerProbLbl.setAttribute("y", marginT + 12);
    markerProbLbl.textContent = `P = ${(p * 100).toFixed(1)}%`;
  }

  updateMarker(probState.currentDate);

  // Drag the marker along the x-axis
  function svgPt(evt) {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }
  let dragging = false;
  const onDown = e => { dragging = true; const p = svgPt(e); updateMarker(xInv(p.x)); };
  const onMove = e => { if (!dragging) return; const p = svgPt(e); updateMarker(xInv(p.x)); };
  const onUp   = () => { dragging = false; };
  svg.addEventListener("mousedown", onDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  svg.style.cursor = "ew-resize";

  wrap.appendChild(svg);

  // Wire input fields ↔ marker
  refs.probDate.oninput = () => {
    const v = parseFloat(refs.probDate.value);
    if (Number.isFinite(v)) updateMarker(v);
  };
  refs.probValue.oninput = () => {
    let v = parseFloat(refs.probValue.value);
    if (!Number.isFinite(v)) return;
    v = Math.max(0.0001, Math.min(0.9999, v));
    updateMarker(normInvCdf(v, mean, sigma));
  };
}


/* ---------- 9. Results table + CSV ---------- */

function renderTable(res) {
  const tbl = refs.resultsTable;
  tbl.innerHTML = "";
  const cols = ["id", "ES", "EF", "LS", "LF", "Slack"];
  const head = document.createElement("thead");
  head.innerHTML = "<tr>" + cols.map(c =>
    `<th class="${c === "id" ? "" : "num"}">${c === "id" ? "Task" : c}</th>`).join("") + "</tr>";
  tbl.appendChild(head);

  const body = document.createElement("tbody");
  const rows = res.tasks.slice().sort((a, b) => (a.ES - b.ES) || a.id.localeCompare(b.id));
  rows.forEach(r => {
    const tr = document.createElement("tr");
    if (r.critical) tr.classList.add("is-critical");
    tr.innerHTML = cols.map((c, i) => {
      if (c === "id") return `<td class="col-id-cell">${r.id}</td>`;
      return `<td class="num">${r[c] === null || r[c] === undefined ? "—" : fmt(r[c], 4)}</td>`;
    }).join("");
    body.appendChild(tr);
  });
  tbl.appendChild(body);
}

function downloadCsv() {
  if (!state.result) return;
  const cols = ["id", "ES", "EF", "LS", "LF", "Slack", "critical"];
  const lines = [cols.join(",")];
  state.result.tasks.forEach(r => {
    lines.push(cols.map(c => r[c] === null || r[c] === undefined ? "" : r[c]).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `pycritical_${state.method}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}


/* ---------- 10. Tooltip ---------- */

function showTooltipForTask(evt, task) {
  const tip = refs.tooltip;
  const dur = (task.EF !== undefined && task.ES !== undefined) ? task.EF - task.ES : null;
  const flag = task.critical
    ? `<span class="tt-flag tt-flag--crit">Critical</span>`
    : `<span class="tt-flag tt-flag--norm">Slack ${fmt(task.Slack, 3)}</span>`;
  tip.innerHTML = `
    <h4>${task.id}</h4>
    <div class="tt-row"><span>Duration</span><b>${fmt(dur, 3)}</b></div>
    <div class="tt-row"><span>Earliest Start</span><b>${fmt(task.ES, 3)}</b></div>
    <div class="tt-row"><span>Earliest Finish</span><b>${fmt(task.EF, 3)}</b></div>
    <div class="tt-row"><span>Latest Start</span><b>${fmt(task.LS, 3)}</b></div>
    <div class="tt-row"><span>Latest Finish</span><b>${fmt(task.LF, 3)}</b></div>
    <div class="tt-row"><span>Slack</span><b>${fmt(task.Slack, 3)}</b></div>
    ${flag}
  `;
  positionTooltip(evt);
  tip.hidden = false;
  requestAnimationFrame(() => tip.classList.add("is-visible"));
}
function positionTooltip(evt) {
  const tip = refs.tooltip;
  const x = evt.clientX, y = evt.clientY;
  // Default offset right & below; flip if near edge
  const pad = 12;
  let nx = x + pad, ny = y + pad;
  // Wait a frame to know its dims
  requestAnimationFrame(() => {
    const r = tip.getBoundingClientRect();
    if (nx + r.width > window.innerWidth - 8) nx = x - r.width - pad;
    if (ny + r.height > window.innerHeight - 8) ny = y - r.height - pad;
    tip.style.left = `${nx}px`;
    tip.style.top  = `${ny}px`;
  });
  tip.style.left = `${nx}px`;
  tip.style.top  = `${ny}px`;
}
function hideTooltip() {
  refs.tooltip.classList.remove("is-visible");
  setTimeout(() => { if (!refs.tooltip.classList.contains("is-visible")) refs.tooltip.hidden = true; }, 160);
}


/* ---------- 11. Bootstrap ---------- */

async function loadExample() {
  refs.datasetPaste.value = EXAMPLE_TEXT[state.method];
  state.pasteDirty = true;
  setPasteStatus("Example loaded. Parsing it in the background while keeping the pasted text visible first…", "info");
  setEditorMode("paste");
  const ok = await parseDatasetFromPaste({ switchToManual: false, showOverlay: false });
  setEditorMode("paste");
  if (ok) {
    setPasteStatus("Example loaded and parsed. The pasted dataset stays visible first; switch to Manual Editor to inspect or tweak the rows.", "success");
  }
}
function clearAll() {
  state.tasks = [];
  state.result = null;
  state.pasteDirty = false;
  refs.datasetPaste.value = "";
  renderEditorBody();
  showEmpty();
  hideError();
  updateKpiStrip(null);
  setPasteStatus("Supports Python-style datasets with optional dataset = and tab-separated spreadsheet paste from Excel or Google Sheets. The schema is auto-detected.", "info");
  setEditorMode("paste");
}

function bind() {
  refs.methodTabs.forEach(b => b.addEventListener("click", () => setMethod(b.dataset.method)));
  refs.editorModeTabs.forEach(b => b.addEventListener("click", () => setEditorMode(b.dataset.editorMode)));
  refs.vizTabs.forEach(b => b.addEventListener("click", () => setView(b.dataset.view)));
  refs.btnAdd.addEventListener("click", () => {
    state.tasks.push(emptyTaskForMethod());
    renderEditorBody();
    setEditorMode("manual");
  });
  refs.datasetPaste.addEventListener("input", () => {
    state.pasteDirty = true;
    setPasteStatus("Paste updated. Parse it to detect the schema and populate the editor.", "info");
  });
  refs.btnParsePaste.addEventListener("click", () => parseDatasetFromPaste({ switchToManual: true }));
  refs.btnRun.addEventListener("click", compute);
  refs.btnLoad.addEventListener("click", loadExample);
  refs.btnLoad2.addEventListener("click", loadExample);
  refs.btnClear.addEventListener("click", clearAll);
  refs.btnDownloadCsv.addEventListener("click", downloadCsv);
  refs.ganttShowSlack.addEventListener("change", () => state.result && renderGantt(state.result));
  refs.ganttShowDeps.addEventListener("change",  () => state.result && renderGantt(state.result));
  refs.netRelayout.addEventListener("click", () => state.result && renderNetwork(state.result));
  window.addEventListener("resize", () => {
    if (!state.result) return;
    if (state.view === "gantt")       renderGantt(state.result);
    if (state.view === "probability") renderProbability(state.result);
  });
}

async function init() {
  bind();
  state.method = "cpm";
  syncMethodUi();
  renderEditor();
  setEditorMode("paste");
  refs.datasetPaste.value = EXAMPLE_TEXT.cpm;
  state.pasteDirty = true;
  state.tasks = [];
  state.result = null;
  showEmpty();
  hideError();
  updateKpiStrip(null);
  setPasteStatus("Example loaded. Parsing it into the manual editor in the background…", "info");
  const ok = await parseDatasetFromPaste({ switchToManual: false, showOverlay: false });
  setEditorMode("paste");
  if (!ok) {
    setPasteStatus("The startup example could not be parsed. Review the pasted text and parse it manually.", "error");
  } else {
    setPasteStatus("Example loaded and parsed. The pasted dataset stays visible first; switch to Manual Editor to review the rows.", "success");
  }
}

document.addEventListener("DOMContentLoaded", init);
