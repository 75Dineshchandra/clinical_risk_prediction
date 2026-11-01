/* Readmission Risk Console: behaviour. Talks only to the API that serves it (same origin). */
"use strict";
(() => {
/* ---------- helpers ---------- */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k === "style") n.style.cssText = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}
const put = (node, ...kids) => node.replaceChildren(...kids.flat(Infinity).filter(k => k !== null && k !== undefined && k !== false));
const when = iso => { if (!iso) return "–"; const d = new Date(String(iso).replace(" ", "T")); return isNaN(d) ? String(iso) : d.toLocaleString(); };
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const fmtMs = s => s === null ? "–" : (s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`);
const ago = sec => sec < 90 ? `${Math.round(sec)} s ago` : sec < 5400 ? `${Math.round(sec / 60)} min ago` : sec < 129600 ? `${(sec / 3600).toFixed(1)} h ago` : `${Math.round(sec / 86400)} days ago`;
const sum = (a, f = x => x) => a.reduce((t, x) => t + f(x), 0);

async function request(path, opts) {
  let res;
  try { res = await fetch(path, opts); } catch (e) { throw { status: 0, body: { message: "Cannot reach the API." } }; }
  const text = await res.text();
  let body = text;
  try { body = JSON.parse(text); } catch (e) { /* plain text, e.g. /metrics */ }
  if (!res.ok) throw { status: res.status, body };
  return body;
}
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => { t.hidden = true; }, 3200);
}
function copy(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(() => toast("Copied"), () => toast("Copy failed"));
  else toast("Copy is not available in this browser");
}
function setPill(node, tone, text) { node.className = `pill ${tone}`; node.textContent = text; }

/* ---------- labels and groups ---------- */
const LABELS = {
  race: "Race", gender: "Gender", age: "Age group", admission_type_id: "Admission type", admission_source_id: "Admission source",
  discharge_disposition_id: "Discharge disposition", payer_code: "Payer", medical_specialty: "Medical specialty",
  time_in_hospital: "Days in hospital", num_lab_procedures: "Lab procedures", num_procedures: "Procedures",
  num_medications: "Medications given", number_outpatient: "Prior outpatient visits", number_emergency: "Prior emergency visits",
  number_inpatient: "Prior inpatient stays", number_diagnoses: "Number of diagnoses", max_glu_serum: "Max glucose serum",
  A1Cresult: "A1C result", change: "Medication changed", diabetesMed: "Diabetes medication", n_drugs_taken: "Drugs taken",
  n_drugs_changed: "Drugs changed", diag_1_group: "Primary diagnosis", diag_2_group: "Secondary diagnosis", diag_3_group: "Additional diagnosis"
};
const labelFor = c => LABELS[c] || c.replace(/_/g, " ").replace(/^./, m => m.toUpperCase());
const MEDS = ["metformin","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide","glipizide","glyburide","tolbutamide",
  "pioglitazone","rosiglitazone","acarbose","miglitol","troglitazone","tolazamide","insulin","glyburide-metformin","glipizide-metformin",
  "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone","change","diabetesMed","n_drugs_taken","n_drugs_changed"];
const GROUPS = [
  { title: "Patient", cols: ["age", "gender", "race"], open: true },
  { title: "Admission and discharge", cols: ["admission_type_id", "admission_source_id", "discharge_disposition_id", "payer_code", "medical_specialty", "time_in_hospital"], open: true },
  { title: "Care during the stay", cols: ["num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses", "max_glu_serum", "A1Cresult"], open: true },
  { title: "Prior hospital use", cols: ["number_outpatient", "number_emergency", "number_inpatient"], open: true },
  { title: "Diagnoses", cols: ["diag_1_group", "diag_2_group", "diag_3_group"], open: true },
  { title: "Medications", cols: MEDS, open: false }
];

const S = { info: null, fields: {}, lastReq: null, lastResp: null, batch: null, cancel: false, monTimer: null };

/* ---------- tabs and status ---------- */
function showView(name) {
  if (!$("#view-" + name)) name = "score";
  $$(".view").forEach(v => { v.hidden = v.id !== "view-" + name; });
  $$("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.view === name));
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  stopMonitor();
  if (name === "monitor") startMonitor();
}
async function pollHealth() {
  const pill = $("#status");
  try { const h = await request("/health"); setPill(pill, "good", `Live · version ${h.model_version} · ${h.model_family}`); }
  catch (e) { setPill(pill, "bad", e.status === 503 ? "No model loaded" : "API unreachable"); }
}

/* ---------- form ---------- */
function makeField(col) {
  const f = S.info.features, isNum = f.numeric.includes(col), id = "f_" + col;
  const wrap = el("label", { class: "field", for: id }, el("span", { class: "lab", title: col, text: labelFor(col) }));
  let get, set;
  if (isNum) {
    const input = el("input", { id, type: "number", step: "any", inputmode: "decimal" });
    input.addEventListener("input", () => wrap.classList.remove("invalid"));
    get = () => input.value.trim() === "" ? null : Number(input.value);
    set = v => { input.value = (v === null || v === undefined) ? "" : v; };
    wrap.append(input);
  } else {
    const levels = (f.categories && f.categories[col]) || [];
    const sel = el("select", { id }, el("option", { value: "", text: "Select…" }),
      levels.map(l => el("option", { value: String(l), text: String(l) })),
      el("option", { value: "__other__", text: "Other (not seen in training)…" }));
    const other = el("input", { type: "text", class: "other", placeholder: "type a value", hidden: true });
    sel.addEventListener("change", () => { wrap.classList.remove("invalid"); other.hidden = sel.value !== "__other__"; if (!other.hidden) other.focus(); });
    other.addEventListener("input", () => wrap.classList.remove("invalid"));
    get = () => sel.value === "" ? null : (sel.value === "__other__" ? (other.value.trim() || null) : sel.value);
    set = v => {
      if (v === null || v === undefined || v === "") { sel.value = ""; other.hidden = true; other.value = ""; return; }
      const s = String(v);
      if (levels.map(String).includes(s)) { sel.value = s; other.hidden = true; }
      else { sel.value = "__other__"; other.hidden = false; other.value = s; }
    };
    wrap.append(sel, other);
  }
  S.fields[col] = { get, set, wrap };
  return wrap;
}
function buildForm() {
  const req = S.info.features.required, mount = $("#formMount");
  mount.replaceChildren(); mount.classList.remove("muted"); S.fields = {};
  const used = new Set();
  const groups = GROUPS.map(g => ({ ...g, cols: g.cols.filter(c => req.includes(c)) }));
  groups.forEach(g => g.cols.forEach(c => used.add(c)));
  const rest = req.filter(c => !used.has(c));
  if (rest.length) groups.push({ title: "Other fields", cols: rest, open: true });
  for (const g of groups) {
    if (!g.cols.length) continue;
    mount.append(el("details", { class: "group", open: g.open },
      el("summary", {}, el("span", { class: "gt", text: g.title }), el("span", { class: "muted small", text: `${g.cols.length} fields` })),
      el("div", { class: "grid" }, g.cols.map(makeField))));
  }
}
function fillForm(rec) {
  Object.entries(S.fields).forEach(([c, f]) => f.set(rec[c]));
  const idf = S.info.features.optional_id_field;
  $("#stayId").value = rec[idf] === undefined || rec[idf] === null ? "" : rec[idf];
  clearInvalid();
}
function clearForm() { Object.values(S.fields).forEach(f => f.set(null)); $("#stayId").value = ""; clearInvalid(); }
function clearInvalid() { $$(".field.invalid").forEach(n => n.classList.remove("invalid")); }
function collect() {
  const rec = {};
  S.info.features.required.forEach(c => { rec[c] = S.fields[c].get(); });
  const id = $("#stayId").value.trim();
  if (id) rec[S.info.features.optional_id_field] = id;
  return rec;
}
async function loadExample(quiet) {
  try { const ex = await request("/example"); fillForm(ex.records[0]); if (!quiet) toast("Example loaded"); }
  catch (e) { if (!quiet) toast("No example is available"); }
}

/* ---------- scoring one stay ---------- */
async function score() {
  clearInvalid();
  const body = { records: [collect()], explain: $("#optExplain").checked, top_k: Number($("#optTopK").value) };
  const btn = $("#btnScore"); btn.disabled = true; btn.textContent = "Scoring…";
  try {
    const resp = await request("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    S.lastReq = body; S.lastResp = resp; renderResult(resp);
  } catch (e) { renderError(e); }
  finally { btn.disabled = false; btn.textContent = "Score this stay"; }
}
function sentence(s, cut, flagged, top) {
  if (cut === null || cut === undefined) return `Score ${fmt(s)}. This model has no high-risk cutoff defined.`;
  const diff = s - cut;
  const close = Math.abs(diff) < 0.05 ? " That is close to the line." : "";
  return `${flagged ? "Flagged as high risk" : "Not flagged"}: the score is ${fmt(Math.abs(diff))} ${diff >= 0 ? "above" : "below"} the cutoff of ${fmt(cut)}, `
    + `the score that the top ${Math.round((top || 0.2) * 100)}% of reference stays reach.${close}`;
}
function meter(score, cut) {
  const pos = v => `${Math.max(0, Math.min(1, v)) * 100}%`;
  const flagged = cut !== null && cut !== undefined && score >= cut;
  const track = el("div", { class: "track" }, el("div", { class: "fill" + (flagged ? " bad" : ""), style: `width:${pos(score)}` }));
  if (cut !== null && cut !== undefined) track.append(el("div", { class: "cut", style: `left:${pos(cut)}` }, el("span", { text: `cutoff ${fmt(cut)}` })));
  track.append(el("div", { class: "dot", style: `left:${pos(score)}` }));
  return el("div", { class: "meter" }, track, el("div", { class: "scale" }, el("span", { text: "0" }), el("span", { text: "0.5" }), el("span", { text: "1" })));
}
function drivers(list) {
  const max = Math.max(...list.map(d => Math.abs(d.contribution)), 1e-9);
  const rows = list.map(d => {
    const w = Math.abs(d.contribution) / max * 50, up = d.contribution > 0;
    const val = (d.input_value === null || d.input_value === undefined) ? "" : ` = ${d.input_value}`;
    return el("div", { class: "drv" },
      el("div", { class: "dname" }, el("b", { text: labelFor(d.column) }), el("span", { class: "muted", text: val })),
      el("div", { class: "dbar" }, el("i", { class: up ? "up" : "down", style: up ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%` })),
      el("div", { class: "dval " + (up ? "up" : "down"), text: (up ? "+" : "") + d.contribution.toFixed(2) }));
  });
  return el("div", { class: "drivers" }, el("h3", { text: "What moved this score" }),
    el("div", { class: "legend small" }, el("span", { text: "◀ lowers risk" }), el("span", { text: "raises risk ▶" })), rows,
    el("p", { class: "small muted mt", text: "Each bar is that factor's push compared with an average stay (log-odds units). Bigger bars mattered more for this stay." }));
}
function jsonDetails() {
  const box = (title, obj) => { const txt = JSON.stringify(obj, null, 2);
    return el("div", { class: "jbox" }, el("div", { class: "jhead" }, el("b", { text: title }), el("button", { class: "btn tiny ghost", type: "button", text: "Copy", onclick: () => copy(txt) })), el("pre", { text: txt })); };
  return el("details", { class: "json" }, el("summary", { text: "Show request and response (JSON)" }), box("Request", S.lastReq), box("Response", S.lastResp));
}
function renderResult(resp) {
  const p = resp.predictions[0], cut = resp.flag_rule.cutoff, flagged = p.high_risk, mount = $("#resultMount");
  put(mount,
    el("div", { class: "res-head" },
      el("div", {}, el("div", { class: "muted small", text: p.id ? `Risk score · stay ${p.id}` : "Risk score" }), el("div", { class: "big", text: fmt(p.score) })),
      cut === null || cut === undefined ? null : el("span", { class: "pill big-pill " + (flagged ? "bad" : "good"), text: flagged ? "High risk" : "Not flagged" })),
    meter(p.score, cut),
    el("p", { class: "explain", text: sentence(p.score, cut, flagged, resp.flag_rule.top_fraction) }),
    p.top_drivers ? drivers(p.top_drivers) : null,
    resp.explain_error ? el("p", { class: "small muted mt", text: `Drivers are unavailable: ${resp.explain_error}` }) : null,
    Object.keys(resp.unseen_categories || {}).length ? el("p", { class: "small mt", style: "color:var(--warn)", text: "Some values were not seen in training: " + Object.entries(resp.unseen_categories).map(([c, v]) => `${labelFor(c)} (${v.join(", ")})`).join("; ") + ". They were scored as 'infrequent'." }) : null,
    el("p", { class: "note small muted", text: resp.score_note }),
    jsonDetails());
}
const ISSUE = { missing: "is missing", null: "is empty", not_numeric: "must be a number" };
function renderError(e) {
  const mount = $("#resultMount"); const d = e.body && e.body.detail !== undefined ? e.body.detail : e.body;
  let title = "Something went wrong"; const lines = []; let firstBad = null;
  if (e.status === 0) title = "Cannot reach the API";
  else if (e.status === 503) { title = "No model is loaded"; lines.push("Export the champion (python serving/export_champion.py), then press Reload on the Model tab."); }
  else if (e.status === 422 && d && d.problems) {
    title = d.message || "The request was rejected";
    d.problems.forEach(p => {
      lines.push(`${labelFor(p.column)} ${ISSUE[p.issue] || p.issue}`);
      const f = S.fields[p.column]; if (f) { f.wrap.classList.add("invalid"); const g = f.wrap.closest("details"); if (g) g.open = true; firstBad = firstBad || f.wrap; }
    });
  } else if (e.status === 422 && Array.isArray(d)) { title = "The request was rejected"; d.forEach(x => lines.push(String(x.msg))); }
  else if (typeof d === "string") lines.push(d);
  else if (d && d.message) lines.push(d.message);
  mount.replaceChildren(el("div", { class: "err" }, el("div", { class: "err-t", text: title }), lines.length ? el("ul", {}, lines.map(l => el("li", { text: l }))) : null));
  if (firstBad && firstBad.scrollIntoView) firstBad.scrollIntoView({ block: "center", behavior: "smooth" });
}

/* ---------- batch ---------- */
function parseCSV(text) {
  const rows = []; let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (c !== "\r") field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  return rows.filter(r => !(r.length === 1 && r[0] === ""));
}
function handleFile(file) {
  if (!file) return;
  $("#dropName").textContent = file.name;
  const reader = new FileReader();
  reader.onload = () => loadCSV(file.name, String(reader.result));
  reader.onerror = () => toast("Could not read that file");
  reader.readAsText(file);
}
function loadCSV(name, text) {
  const rows = parseCSV(text);
  if (rows.length < 2) { $("#batchInfo").replaceChildren(el("div", { class: "err" }, el("div", { class: "err-t", text: "The file has no data rows" }))); $("#batchActions").hidden = true; return; }
  const header = rows.shift().map(h => h.trim()), f = S.info.features, idCol = f.optional_id_field;
  const idx = {}; header.forEach((h, i) => { idx[h] = i; });
  const missing = f.required.filter(c => !(c in idx));
  const ignored = header.filter(h => !f.required.includes(h) && h !== idCol);
  S.batch = { name, rows, idx };
  const info = el("div", {},
    el("div", { class: "row" }, el("span", { class: "pill nodot", text: name }), el("span", { text: `${rows.length.toLocaleString()} rows · ${header.length} columns` })),
    missing.length ? el("div", { class: "err mt" }, el("div", { class: "err-t", text: `${missing.length} required column${missing.length > 1 ? "s are" : " is"} missing` }), el("div", { class: "chips" }, missing.map(m => el("span", { class: "chip", text: m })))) : el("p", { class: "small muted mt", text: `All ${f.required.length} required columns are present.` }),
    ignored.length ? el("p", { class: "small muted mt", text: `${ignored.length} other column${ignored.length > 1 ? "s" : ""} will be ignored (including any outcome columns): ${ignored.slice(0, 8).join(", ")}${ignored.length > 8 ? "…" : ""}` }) : null);
  $("#batchInfo").replaceChildren(info);
  $("#batchN").value = Math.min(rows.length, 5000); $("#batchN").max = rows.length; $("#batchOf").textContent = `of ${rows.length.toLocaleString()} (the most is scored in 500-row requests)`;
  $("#batchActions").hidden = false; $("#btnBatch").disabled = missing.length > 0;
  $("#batchResults").replaceChildren();
}
function rowToRecord(r) {
  const B = S.batch, f = S.info.features, numeric = new Set(f.numeric), rec = {};
  f.required.forEach(c => { const v = (r[B.idx[c]] ?? "").trim(); rec[c] = v === "" ? null : (numeric.has(c) ? Number(v) : v); });
  const idCol = f.optional_id_field;
  if (idCol in B.idx) rec[idCol] = r[B.idx[idCol]];
  return rec;
}
async function runBatch() {
  const B = S.batch, max = S.info.limits.max_records_per_request, explain = $("#batchExplain").checked;
  const total = Math.max(1, Math.min(Number($("#batchN").value) || B.rows.length, B.rows.length));
  const results = []; let last = null; S.cancel = false;
  $("#btnBatch").disabled = true; $("#btnCancel").hidden = false; $("#progress").hidden = false; $("#bar").style.width = "0%"; $("#batchResults").replaceChildren();
  for (let start = 0; start < total && !S.cancel; start += max) {
    const chunk = B.rows.slice(start, Math.min(start + max, total)).map(rowToRecord);
    let resp;
    try { resp = await request("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ records: chunk, explain, top_k: 3 }) }); }
    catch (e) { batchError(e, start); break; }
    last = resp;
    resp.predictions.forEach((p, i) => results.push({ row: start + i, id: p.id, score: p.score, high_risk: p.high_risk, tops: p.top_drivers || null }));
    $("#bar").style.width = `${Math.min(100, results.length / total * 100)}%`;
  }
  $("#btnBatch").disabled = false; $("#btnCancel").hidden = true;
  if (results.length && last) renderBatch(results, last.flag_rule, S.cancel || results.length < total);
}
function batchError(e, start) {
  const d = e.body && e.body.detail !== undefined ? e.body.detail : e.body; const lines = [];
  if (d && d.problems) d.problems.forEach(p => lines.push(`${labelFor(p.column)} ${ISSUE[p.issue] || p.issue}` + (p.rows ? ` (file rows ${p.rows.map(r => start + r + 2).join(", ")})` : "")));
  else lines.push(typeof d === "string" ? d : (d && d.message) || `HTTP ${e.status}`);
  $("#batchResults").replaceChildren(el("div", { class: "err" }, el("div", { class: "err-t", text: (d && d.message) || "The file was rejected" }), el("ul", {}, lines.map(l => el("li", { text: l })))));
}
function histogram(values, cut, bins = 20) {
  const counts = new Array(bins).fill(0);
  values.forEach(v => { counts[Math.min(bins - 1, Math.max(0, Math.floor(v * bins)))]++; });
  const max = Math.max(...counts, 1);
  const h = el("div", { class: "hist" }, counts.map((n, i) => el("div", { class: "b", style: `height:${n / max * 100}%`, title: `${(i / bins).toFixed(2)} to ${((i + 1) / bins).toFixed(2)}: ${n}` })));
  if (cut !== null && cut !== undefined) h.append(el("div", { class: "cutline", style: `left:${Math.max(0, Math.min(1, cut)) * 100}%`, title: `cutoff ${fmt(cut)}` }));
  return el("div", {}, h, el("div", { class: "hist-scale" }, el("span", { text: "0" }), el("span", { text: "score" }), el("span", { text: "1" })));
}
function explainRow(i) {
  const rec = rowToRecord(S.batch.rows[i]);
  showView("score"); fillForm(rec);
  try { window.scrollTo(0, 0); } catch (e) { /* not available everywhere */ }
  score();
}
const driverText = t => t ? `${labelFor(t.column)}${t.input_value !== null && t.input_value !== undefined ? " = " + t.input_value : ""} (${t.contribution > 0 ? "+" : ""}${t.contribution.toFixed(2)})` : "";
function renderBatch(results, rule, partial) {
  const n = results.length, flagged = results.filter(r => r.high_risk).length, mean = sum(results, r => r.score) / n;
  const inFile = S.batch.rows.length, hasTop = results.some(r => r.tops && r.tops.length);
  const stats = el("div", { class: "stats" },
    stat("Stays scored", n.toLocaleString(), partial ? "stopped early" : (n < inFile ? `of ${inFile.toLocaleString()} in the file` : "")),
    stat("Flagged high risk", pct(flagged / n), `${flagged.toLocaleString()} stays (expected about ${pct(rule.top_fraction || 0.2, 0)})`),
    stat("Average score", fmt(mean), "ranking score, not a probability"), stat("Cutoff", fmt(rule.cutoff), "high risk if score ≥ cutoff"));
  const order = el("select", { style: "width:auto" }, el("option", { value: "high", text: "Highest scores first" }), el("option", { value: "file", text: "File order" }));
  const search = el("input", { type: "text", placeholder: "Find a stay ID…", style: "width:160px", "aria-label": "Find a stay ID" });
  const tableMount = el("div", { class: "mt" });
  const draw = () => {
    const q = search.value.trim().toLowerCase();
    let rows = q ? results.filter(r => String(r.id === null || r.id === undefined ? "" : r.id).toLowerCase().includes(q)) : results;
    if (order.value === "high") rows = [...rows].sort((a, b) => b.score - a.score);
    const shown = rows.slice(0, 100);
    put(tableMount,
      rows.length ? el("div", { class: "tablewrap" }, el("table", {},
        el("thead", {}, el("tr", {}, el("th", { text: "Stay" }), el("th", { class: "num", text: "Score" }), el("th", { text: "Flag" }), hasTop ? el("th", { text: "Top driver" }) : null)),
        el("tbody", {}, shown.map(r => el("tr", { class: "clickable", title: "Click to see why this stay scored this way", onclick: () => explainRow(r.row) },
          el("td", { text: r.id === null || r.id === undefined ? "row " + (r.row + 2) : r.id }),
          el("td", { class: "num" }, fmt(r.score), el("span", { class: "minibar", style: `width:${Math.round(r.score * 60)}px` })),
          el("td", {}, el("span", { class: "pill " + (r.high_risk ? "bad" : "good"), text: r.high_risk ? "High risk" : "Not flagged" })),
          hasTop ? el("td", { class: "small", text: driverText(r.tops && r.tops[0]) }) : null)))))
        : el("p", { class: "muted mt", text: "No stay matches that ID." }),
      rows.length > 100 ? el("p", { class: "small muted mt", text: `Showing ${shown.length} of ${rows.length.toLocaleString()}${q ? " matches" : ""}. Download the CSV for everything.` }) : null,
      rows.length ? el("p", { class: "small muted mt", text: "Click any row to open that stay in “Score a stay” with its full explanation." }) : null);
  };
  order.addEventListener("change", draw); search.addEventListener("input", draw); draw();
  const dl = el("button", { class: "btn ghost", type: "button", text: "Download results (CSV)", onclick: () => downloadResults(results) });
  const card = el("div", { class: "card" });
  card.append(el("div", { class: "card-head" }, el("h2", { text: "Results" }), el("div", { class: "btns" }, search, order, dl)), stats,
    el("h3", { class: "mt2", text: "Score distribution" }), histogram(results.map(r => r.score), rule.cutoff), tableMount);
  $("#batchResults").replaceChildren(card);
}
function stat(k, v, h) { return el("div", { class: "stat" }, el("div", { class: "k", text: k }), el("div", { class: "v", text: v }), h ? el("div", { class: "h", text: h }) : null); }
function downloadResults(results) {
  const q = v => { const s = v === null || v === undefined ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  const hasTop = results.some(r => r.tops && r.tops.length);
  const head = ["file_row", "id", "score", "high_risk"].concat(hasTop ? ["driver_1", "driver_2", "driver_3"] : []);
  const lines = results.map(r => [r.row + 2, r.id, r.score, r.high_risk].concat(hasTop ? [0, 1, 2].map(k => driverText(r.tops && r.tops[k])) : []).map(q).join(","));
  const url = URL.createObjectURL(new Blob([[head.join(",")].concat(lines).join("\n")], { type: "text/csv" }));
  const a = el("a", { href: url, download: "readmission_scores.csv" }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ---------- model tab ---------- */
function kv(rows) { return el("dl", { class: "kv" }, rows.filter(Boolean).map(([k, v]) => [el("dt", { text: k }), el("dd", {}, v)])); }
function renderModel() {
  const i = S.info, m = i.model, ev = i.evidence || {}, b = i.bundle, mount = $("#modelMount");
  const cards = [];
  cards.push(el("div", { class: "card" },
    el("div", { class: "row", style: "justify-content:space-between" }, el("h2", { text: "Live model" }), el("span", { class: "pill good", text: "Serving" })),
    el("p", { class: "mt", style: "font-size:22px;font-weight:720;letter-spacing:-.02em", text: `${m.registered_model} · version ${m.version}` }),
    el("div", { class: "row mt" }, el("span", { class: "pill nodot", text: m.model_family || "model" }), el("span", { class: "pill nodot", text: "@" + (m.alias || "champion") })),
    kv([["Training run", el("span", { class: "mono small", text: m.training_run_id || "–" })], ["Exported", when(b.exported_at)], ["Loaded by the API", when(b.loaded_at)],
        ["Checksum", el("span", { class: "mono small", text: b.model_sha256 || "–" })], ["Python at export", b.python_at_export || "–"]])));
  const metric = (k, key) => ev[key] ? stat(k, ev[key], "") : null;
  cards.push(el("div", { class: "card" },
    el("div", { class: "row", style: "justify-content:space-between" }, el("h2", { text: "Evidence" }), ev.gate ? el("span", { class: "pill " + (ev.gate === "PASS" ? "good" : "bad"), text: "Gate " + ev.gate }) : null),
    el("p", { class: "muted small mt", text: "Recorded in the registry when this model was registered. Scores are on data the model never trained on." }),
    el("div", { class: "stats mt" }, [metric("Test PR-AUC", "test_pr_auc"), metric("Test ROC-AUC", "test_roc_auc"), metric("Lift over chance", "test_lift"), metric("CV PR-AUC", "cv_pr_auc")].filter(Boolean)),
    ev.selection_reason ? el("p", { class: "mt", text: ev.selection_reason }) : null,
    ev.promoted_at ? el("p", { class: "small muted mt", text: `Promoted ${when(ev.promoted_at)}` }) : null));
  const fr = i.flag_rule;
  cards.push(el("div", { class: "card" }, el("h2", { text: "High-risk rule" }),
    el("p", { class: "mt", text: `A stay is flagged when its score is at or above ${fmt(fr.cutoff, 4)}. That is the score reached by the top ${Math.round((fr.top_fraction || 0.2) * 100)}% of ${fr.based_on ? fr.based_on.rows.toLocaleString() : ""} reference stays.` }),
    el("p", { class: "small muted mt", text: i.score_note })));
  const f = i.features;
  cards.push(el("div", { class: "card" }, el("h2", { text: "Inputs" }),
    el("p", { class: "mt", text: `${f.required.length} columns are required: ${f.numeric.length} numbers and ${f.categorical.length} categories.` }),
    el("details", { class: "json" }, el("summary", { text: "Show the column names" }),
      el("div", { class: "chips" }, f.required.map(c => el("span", { class: "chip" + (f.numeric.includes(c) ? "" : ""), title: f.numeric.includes(c) ? "number" : "category", text: c })))),
    el("p", { class: "small muted mt", text: `Optional: an ${f.optional_id_field} is returned as the stay id. Other columns are ignored. Up to ${i.limits.max_records_per_request} stays per request.` })));
  cards.push(el("div", { class: "card" }, el("h2", { text: "Explanations and checks" }),
    kv([["Drivers", i.explanations === "tree" ? "SHAP values (tree model)" : i.explanations === "linear" ? "Coefficients × values (linear model)" : "Not available"]]),
    i.package_warnings && i.package_warnings.length
      ? el("div", { class: "banner mt" }, "Package versions differ from the export: ", i.package_warnings.join("; "))
      : el("p", { class: "small muted mt", text: "Package versions match the ones used when the bundle was exported." })));
  const tokenIn = el("input", { type: "text", id: "adminToken", placeholder: "Admin token", hidden: true, class: "mt" });
  cards.push(el("div", { class: "card" }, el("h2", { text: "Update the model" }),
    el("p", { class: "muted small mt", text: "After a promotion or rollback, run python serving/export_champion.py, then reload here. The old model keeps serving if the new bundle is incomplete." }),
    tokenIn, el("div", { class: "btns mt" }, el("button", { class: "btn primary", type: "button", text: "Reload model", onclick: () => reloadModel(tokenIn) }))));
  mount.replaceChildren(...cards);
}
async function reloadModel(tokenIn) {
  const headers = tokenIn.value.trim() ? { "X-Admin-Token": tokenIn.value.trim() } : {};
  try {
    const r = await request("/reload", { method: "POST", headers });
    toast(r.previous_version === r.model_version ? `Reloaded: still version ${r.model_version}` : `Reloaded: version ${r.previous_version} → ${r.model_version}`);
    await refreshInfo(true); pollHealth();
  } catch (e) {
    if (e.status === 403) { tokenIn.hidden = false; tokenIn.focus(); toast("This API needs an admin token"); }
    else { const d = e.body && e.body.detail; toast(d && d.error ? `Reload failed, still serving version ${d.still_serving_version}: ${d.error}` : "Reload failed"); }
  }
}

/* ---------- monitoring ---------- */
function parseProm(text) {
  const out = [];
  for (const line of String(text).split("\n")) {
    if (!line || line[0] === "#") continue;
    const m = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)$/); if (!m) continue;
    const labels = {};
    if (m[2]) for (const x of m[2].matchAll(/([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"/g)) labels[x[1]] = x[2];
    out.push({ name: m[1], labels, value: Number(m[3]) });
  }
  return out;
}
const total = (P, name, f = () => true) => sum(P.filter(s => s.name === name && f(s.labels)), s => s.value);
function quantileLE(P, name, filter, q) {
  const b = P.filter(s => s.name === name + "_bucket" && filter(s.labels)).map(s => ({ le: s.labels.le === "+Inf" ? Infinity : Number(s.labels.le), c: s.value })).sort((a, b) => a.le - b.le);
  const n = b.length ? b[b.length - 1].c : 0; if (!n) return null;
  const hit = b.find(x => x.c >= q * n); return hit && hit.le !== Infinity ? hit.le : null;
}
async function loadMonitor() {
  let text;
  try { text = await request("/metrics"); } catch (e) { $("#monMount").replaceChildren(el("div", { class: "err" }, el("div", { class: "err-t", text: "Cannot read /metrics" }))); return; }
  const P = parseProm(text), cut = S.info && S.info.flag_rule ? S.info.flag_rule.cutoff : null, top = S.info && S.info.flag_rule ? S.info.flag_rule.top_fraction : 0.2;
  const scored = total(P, "readmission_predictions_total"), high = total(P, "readmission_predictions_total", l => l.outcome === "high_risk");
  const predReq = total(P, "readmission_http_requests_total", l => l.path === "/predict");
  const dCount = total(P, "readmission_http_request_duration_seconds_count", l => l.path === "/predict");
  const dSum = total(P, "readmission_http_request_duration_seconds_sum", l => l.path === "/predict");
  const p95 = quantileLE(P, "readmission_http_request_duration_seconds", l => l.path === "/predict", 0.95);
  const bad = total(P, "readmission_http_requests_total", l => !String(l.status).startsWith("2"));
  const loadedTs = total(P, "readmission_model_loaded_timestamp_seconds"), reloadsOk = total(P, "readmission_model_reloads_total", l => l.outcome === "ok"), reloadsBad = total(P, "readmission_model_reloads_total", l => l.outcome === "failed");
  const unseen = P.filter(s => s.name === "readmission_unseen_category_total" && s.value > 0), verr = P.filter(s => s.name === "readmission_validation_errors_total" && s.value > 0);
  const stats = el("div", { class: "stats" },
    stat("Stays scored", scored.toLocaleString(), `${predReq.toLocaleString()} scoring requests`),
    stat("Flagged high risk", scored ? pct(high / scored) : "–", `expected about ${pct(top || 0.2, 0)} on typical data`),
    stat("Average response", dCount ? fmtMs(dSum / dCount) : "–", "per scoring request"),
    stat("95% faster than", fmtMs(p95), "slowest 5% are above this"),
    stat("Rejected or failed", bad.toLocaleString(), "requests with a non-2xx status"),
    stat("Model loaded", loadedTs ? ago(Date.now() / 1000 - loadedTs) : "–", `${reloadsOk} load${reloadsOk === 1 ? "" : "s"} ok, ${reloadsBad} failed`));
  const sb = P.filter(s => s.name === "readmission_prediction_score_bucket" && s.labels.le !== "+Inf").map(s => ({ le: Number(s.labels.le), c: s.value })).sort((a, b) => a.le - b.le);
  const counts = sb.map((x, i) => ({ lo: i ? sb[i - 1].le : 0, hi: x.le, n: x.c - (i ? sb[i - 1].c : 0) })), max = Math.max(...counts.map(c => c.n), 1);
  const hist = el("div", { class: "hist" }, counts.map(c => el("div", { class: "b", style: `height:${c.n / max * 100}%`, title: `${c.lo.toFixed(2)} to ${c.hi.toFixed(2)}: ${c.n}` })));
  if (cut !== null && cut !== undefined) hist.append(el("div", { class: "cutline", style: `left:${Math.max(0, Math.min(1, cut)) * 100}%`, title: `cutoff ${fmt(cut)}` }));
  const reqRows = P.filter(s => s.name === "readmission_http_requests_total").sort((a, b) => b.value - a.value);
  const listOrNone = (items, render, none) => items.length ? el("div", { class: "chips" }, items.map(render)) : el("p", { class: "small mt", style: "color:var(--good)", text: none });
  $("#monMount").replaceChildren(stats,
    el("div", { class: "card mt" }, el("h3", { text: "Score distribution of everything scored so far" }), scored ? hist : el("p", { class: "muted small mt", text: "Nothing has been scored yet." }),
      el("div", { class: "hist-scale" }, el("span", { text: "0" }), el("span", { text: cut !== null && cut !== undefined ? `line = cutoff ${fmt(cut)}` : "score" }), el("span", { text: "1" }))),
    el("div", { class: "cards mt" },
      el("div", { class: "card" }, el("h3", { text: "Rejected requests, by reason" }), listOrNone(verr, s => el("span", { class: "chip", text: `${s.labels.reason}: ${s.value}` }), "None so far.")),
      el("div", { class: "card" }, el("h3", { text: "Values never seen in training" }), listOrNone(unseen, s => el("span", { class: "chip", text: `${s.labels.column}: ${s.value}` }), "None so far."))),
    el("div", { class: "card mt" }, el("h3", { text: "Requests by endpoint" }),
      el("div", { class: "tablewrap mt" }, el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "Method" }), el("th", { text: "Path" }), el("th", { text: "Status" }), el("th", { class: "num", text: "Count" }))),
        el("tbody", {}, reqRows.map(s => el("tr", {}, el("td", { text: s.labels.method }), el("td", { class: "mono small", text: s.labels.path }),
          el("td", {}, el("span", { class: "pill nodot " + (String(s.labels.status).startsWith("2") ? "good" : "bad"), text: s.labels.status })), el("td", { class: "num", text: s.value.toLocaleString() }))))))));
}
function startMonitor() { loadMonitor(); if ($("#autoRefresh").checked) S.monTimer = setInterval(loadMonitor, 5000); }
function stopMonitor() { clearInterval(S.monTimer); S.monTimer = null; }

/* ---------- start-up ---------- */
async function refreshInfo(rebuild) {
  const before = S.info ? JSON.stringify(S.info.features.required) : null;
  try { S.info = await request("/model-info"); $("#banner").hidden = true; }
  catch (e) {
    const b = $("#banner"); b.hidden = false;
    b.textContent = e.status === 503 ? "No model is loaded yet. Run python serving/export_champion.py, then press Reload on the Model tab." : "Cannot reach the API.";
    return false;
  }
  if (rebuild || before !== JSON.stringify(S.info.features.required)) { buildForm(); await loadExample(true); }
  renderModel(); return true;
}
const UI_VERSION = "1.2";
async function init() {
  $("#uiver").textContent = `Console version ${UI_VERSION}.`;
  $$("#tabs button").forEach(b => b.addEventListener("click", () => showView(b.dataset.view)));
  $("#btnExample").addEventListener("click", () => loadExample(false));
  $("#btnClear").addEventListener("click", clearForm);
  $("#btnScore").addEventListener("click", score);
  $("#formMount").addEventListener("keydown", e => { if (e.key === "Enter" && e.target.tagName !== "SELECT") { e.preventDefault(); score(); } });
  const drop = $("#drop"), file = $("#csvFile");
  drop.addEventListener("click", () => file.click());
  file.addEventListener("change", () => handleFile(file.files && file.files[0]));
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => handleFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]));
  $("#btnBatch").addEventListener("click", runBatch);
  $("#btnCancel").addEventListener("click", () => { S.cancel = true; });
  $("#btnRefreshMon").addEventListener("click", loadMonitor);
  $("#autoRefresh").addEventListener("change", () => { stopMonitor(); if (!$("#view-monitor").hidden) startMonitor(); });
  showView((location.hash || "#score").slice(1));
  pollHealth(); setInterval(pollHealth, 15000);
  await refreshInfo(true);
}
init();
})();