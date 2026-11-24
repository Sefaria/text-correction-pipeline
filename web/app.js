// Dashboard: start projects, watch their progress, open / export / rename / archive / delete them.
import { api, esc, postJSON, progressHTML, toast } from "./common.js";

const $ = (s) => document.querySelector(s);
let projects = [];
let pollTimer;

// ---- book autocomplete + versions ----------------------------------------------------------------------------------------------------
const book = $("#book"), list = $("#book-list"), version = $("#version");
let suggestTimer, active = -1, chosenBook = "";

book.addEventListener("input", () => {
  clearTimeout(suggestTimer);
  chosenBook = "";
  version.disabled = true;
  version.innerHTML = `<option value="">Choose a book first</option>`;
  suggestTimer = setTimeout(async () => {
    const q = book.value.trim();
    if (q.length < 2) return hideList();
    const hits = await api(`/api/books?q=${encodeURIComponent(q)}`);
    active = -1;
    list.innerHTML = hits.map((t, i) => `<li role="option" id="bk-${i}" data-t="${esc(t)}">${esc(t)}</li>`).join("");
    list.hidden = !hits.length;
    book.setAttribute("aria-expanded", String(!!hits.length));
  }, 180);
});
book.addEventListener("keydown", (e) => {
  const items = [...list.querySelectorAll("li")];
  if (list.hidden || !items.length) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
    items.forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
    book.setAttribute("aria-activedescendant", items[active].id);
  } else if (e.key === "Enter" && active >= 0) {
    e.preventDefault();
    chooseBook(items[active].dataset.t);
  } else if (e.key === "Escape") hideList();
});
list.addEventListener("mousedown", (e) => {
  const li = e.target.closest("li");
  if (li) chooseBook(li.dataset.t);
});
book.addEventListener("blur", () => setTimeout(hideList, 150));
function hideList() { list.hidden = true; book.setAttribute("aria-expanded", "false"); }

async function chooseBook(t) {
  book.value = chosenBook = t;
  hideList();
  version.innerHTML = `<option>Loading…</option>`;
  const vs = await api(`/api/versions?title=${encodeURIComponent(t)}`);
  if (!vs.length) {
    version.innerHTML = `<option value="">No Hebrew versions</option>`;
    return;
  }
  version.innerHTML = vs.map((v, i) =>
    `<option value="${esc(v.versionTitle)}" data-license="${esc(v.license)}">${esc(v.versionTitle)}${i === 0 ? " (default)" : ""}</option>`).join("");
  version.disabled = false;
  showVersionInfo();
}
version.addEventListener("change", showVersionInfo);
function showVersionInfo() {
  const opt = version.selectedOptions[0];
  $("#version-info").textContent = opt?.dataset.license ? `License: ${opt.dataset.license}` : "";
}

// ---- create --------------------------------------------------------------------------------------------------------------------------
$("#new").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!chosenBook) return toast("Choose a book from the list");
  if (!version.value) return toast("Choose a version");
  const fd = new FormData();
  fd.append("title", chosenBook);
  fd.append("version_title", version.value);
  fd.append("name", $("#name").value);
  if ($("#threshold").value) fd.append("threshold", $("#threshold").value);
  if ($("#llm").value.trim()) fd.append("llm", $("#llm").value.trim());
  const f = $("#reference").files[0];
  if (f) fd.append("reference", f);
  $("#create").disabled = true;
  try {
    await api("/api/projects", { method: "POST", body: fd });
    toast("Project queued");
    $("#new").reset();
    chosenBook = "";
    version.disabled = true;
    await load();
  } catch (err) {
    toast(err.message, 5000);
  } finally {
    $("#create").disabled = false;
  }
});

// ---- list ----------------------------------------------------------------------------------------------------------------------------
function ago(t) {
  const s = Date.now() / 1000 - t;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
}

function render() {
  const tb = $("#projects");
  $("#no-projects").hidden = projects.length > 0;
  tb.innerHTML = projects.map((p) => {
    const c = p.counts, pr = p.progress;
    const running = pr.status === "running" || pr.status === "queued";
    const cost = p.usage?.cost_usd != null ? ` · LLM $${p.usage.cost_usd.toFixed(2)}` : "";
    return `<tr data-id="${p.id}">
      <td><a href="/review?id=${p.id}"><b>${esc(p.name)}</b></a><div class="small muted">${esc(p.owner)}</div></td>
      <td>${esc(p.title)}<div class="small muted">${esc(p.version_title)}${p.words ? ` · ${p.words.toLocaleString()} words` : ""}</div></td>
      <td class="small">${p.ref_filename ? esc(p.ref_filename) : `<span class="muted">—</span>`}</td>
      <td>${progressHTML(pr)}</td>
      <td class="small">${c.todo} to review<br><span class="muted">${c.accepted} accepted · ${c.rejected} rejected · ${c.overridden} edited${cost}</span></td>
      <td class="small muted">${ago(p.updated_at)}</td>
      <td><div class="actions">
        <button data-a="open">Open</button>
        <button data-a="export" ${c.changed_segments ? "" : "disabled"} title="${c.changed_segments} changed segments">Export CSV</button>
        <button data-a="rename">Rename</button>
        ${running ? `<button data-a="cancel">Cancel</button>` : ""}
        ${pr.status === "failed" || pr.status === "cancelled" ? `<button data-a="retry">Retry</button>` : ""}
        <button data-a="archive">${p.archived ? "Unarchive" : "Archive"}</button>
        ${running ? "" : `<button data-a="delete" class="danger">Delete</button>`}
      </div></td></tr>`;
  }).join("");
}

$("#projects").addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  const id = +b.closest("tr").dataset.id, p = projects.find((x) => x.id === id);
  try {
    switch (b.dataset.a) {
      case "open": location.href = `/review?id=${id}`; return;
      case "export": location.href = `/api/projects/${id}/export.csv`; return;
      case "rename": {
        const cell = b.closest("tr").querySelector("td");
        cell.innerHTML = `<form class="row" data-rename><input aria-label="Project name" value="${esc(p.name)}" style="flex:1">
          <button class="primary">Save</button></form>`;
        clearTimeout(pollTimer);                         // don't re-render while editing
        const input = cell.querySelector("input");
        input.focus(); input.select();
        input.addEventListener("keydown", (ev) => { if (ev.key === "Escape") load(); });
        cell.querySelector("form").addEventListener("submit", async (ev) => {
          ev.preventDefault();
          if (input.value.trim()) await postJSON(`/api/projects/${id}`, { name: input.value.trim() }, "PATCH");
          load();
        });
        return;
      }
      case "cancel": await api(`/api/projects/${id}/cancel`, { method: "POST" }); break;
      case "retry": await api(`/api/projects/${id}/retry`, { method: "POST" }); break;
      case "archive": await postJSON(`/api/projects/${id}`, { archived: !p.archived }, "PATCH"); break;
      case "delete":                                     // two clicks: the first arms the button
        if (b.dataset.armed !== "1") {
          b.dataset.armed = "1";
          b.textContent = "Confirm delete";
          clearTimeout(pollTimer);
          setTimeout(load, 4000);
          return;
        }
        await api(`/api/projects/${id}`, { method: "DELETE" });
        toast(`Deleted "${p.name}"`);
        break;
    }
    await load();
  } catch (err) {
    toast(err.message, 5000);
  }
});

async function load() {
  const qs = new URLSearchParams({ all: $("#all").checked, archived: $("#archived").checked });
  projects = await api(`/api/projects?${qs}`);
  render();
  clearTimeout(pollTimer);
  const busy = projects.some((p) => ["running", "queued"].includes(p.progress.status));
  pollTimer = setTimeout(load, busy ? 2000 : 15000);
}
$("#all").addEventListener("change", load);
$("#archived").addEventListener("change", load);

(async () => {
  const me = await api("/api/me");
  $("#me").textContent = me.email;
  await load();
})();
