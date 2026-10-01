// Dashboard del cluster PDN. Sin framework: estado por WebSocket y API REST.
// Regla: un valor que no se midio se muestra como "sin dato", nunca como cero.
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const COLORES = { cpu: "#1f6feb", gpu: "#1a7f37", npu: "#8250df", opencl: "#bf8700" };
const est = { estado: null, pestana: "topologia", serieMbs: [], cpuHist: {}, graficos: {}, firmaNodos: "",
  resultadoSel: null, ultimaCorrida: null, sinConexion: 0 };

// ---------------------------------------------------------------- utilidades
function esc(x) {
  return String(x).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function sd(motivo) { return `<span class="sin-dato"${motivo ? ` title="${esc(motivo)}"` : ""}>sin dato</span>`; }
function num(x, dec = 1, unidad = "") {
  if (x === null || x === undefined || Number.isNaN(x)) return sd();
  const v = Number(x);
  return v.toLocaleString("es-EC", { maximumFractionDigits: dec, minimumFractionDigits: 0 }) + (unidad ? " " + unidad : "");
}
function bytes(b) {
  if (b === null || b === undefined) return sd();
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; let v = Number(b);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return v.toLocaleString("es-EC", { maximumFractionDigits: i ? 2 : 0 }) + " " + u[i];
}
function hora(t) { return t ? new Date(t * 1000).toLocaleTimeString("es-EC") : sd(); }
function dl(pares) { return pares.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join(""); }
async function api(metodo, ruta, cuerpo) {
  const r = await fetch(ruta, { method: metodo, headers: { "Content-Type": "application/json" },
    body: cuerpo === undefined ? undefined : JSON.stringify(cuerpo) });
  const t = await r.text();
  let d; try { d = JSON.parse(t); } catch { d = t; }
  if (!r.ok) throw new Error((d && d.detail) || t || r.statusText);
  return d;
}
function grafico(id, tipo, datos, opciones = {}) {
  const lienzo = document.getElementById(id);
  if (!lienzo || typeof Chart === "undefined") return null;
  if (est.graficos[id]) { est.graficos[id].data = datos; est.graficos[id].update("none"); return est.graficos[id]; }
  est.graficos[id] = new Chart(lienzo, { type: tipo, data: datos,
    options: Object.assign({ animation: false, responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { boxWidth: 12 } } } }, opciones) });
  return est.graficos[id];
}
function porNodo(workers) {
  const nodos = {};
  for (const w of workers) (nodos[w.hostname] = nodos[w.hostname] || []).push(w);
  return nodos;
}

// ---------------------------------------------------------------- pestanas
$$("#pestanas button").forEach(b => b.addEventListener("click", () => {
  $$("#pestanas button").forEach(x => x.classList.toggle("activa", x === b));
  $$(".pantalla").forEach(s => s.classList.toggle("activa", s.id === b.dataset.p));
  est.pestana = b.dataset.p;
  // Los graficos creados con la pestana oculta quedan con tamano cero
  requestAnimationFrame(() => Object.values(est.graficos).forEach(g => g.resize()));
  if (est.pestana === "resultados") cargarResultados();
  if (est.pestana === "evidencias") cargarEvidencias();
  if (est.pestana === "registro") cargarRegistro();
  if (est.pestana === "escalabilidad") cargarEscalabilidad();
  if (est.estado) pintar(est.estado);
}));

// ---------------------------------------------------------------- conexion
function conectar() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onmessage = ev => { est.sinConexion = 0; recibir(JSON.parse(ev.data)); };
  ws.onclose = () => { est.sinConexion++; avisoConexion(); setTimeout(conectar, 2000); };
  ws.onerror = () => ws.close();
}
function avisoConexion() {
  const a = $("#aviso-respaldo");
  if (est.sinConexion < 2) return;
  const resp = ((est.estado || {}).nodos_config || []).find(n => (n.roles || []).includes("respaldo"));
  a.classList.remove("oculto");
  a.textContent = "El Master no responde." + (resp ? ` Si el respaldo tomó el control, abra http://${resp.ip}:${location.port || 8000}` : "");
}
function recibir(e) {
  est.estado = e;
  const a = $("#aviso-respaldo");
  if (e.rol === "respaldo") {
    a.classList.remove("oculto");
    a.textContent = "Master de respaldo activo desde " + hora(e.activo_desde).replace(/<[^>]+>/g, "");
  } else a.classList.add("oculto");
  const conectados = e.workers.filter(w => w.estado === "conectado").length;
  $("#estado-master").innerHTML = `Master ${esc(e.rol)} · puerto ${e.puerto} · ${conectados}/${e.workers.length} workers conectados` +
    (e.respaldo_conectado ? " · respaldo conectado" : "");
  const c = e.corrida;
  if (c && c.tiempo_s !== null && c.mb_s !== null) {
    if (est.ultimaCorrida !== c.corrida_id) { est.serieMbs = []; est.ultimaCorrida = c.corrida_id; }
    if (c.estado === "EJECUTANDO") est.serieMbs.push({ x: c.tiempo_s, y: c.mb_s });
  }
  for (const w of e.workers) {
    const m = w.metricas || {};
    const h = (est.cpuHist[w.wid] = est.cpuHist[w.wid] || []);
    if (m.t && (!h.length || h[h.length - 1].t !== m.t)) h.push({ t: m.t, y: m.cpu_pct });
    if (h.length > 120) h.shift();
  }
  pintar(e);
}
function pintar(e) {
  if (est.pestana === "topologia") pintarTopologia(e);
  if (est.pestana === "configuracion") pintarConfiguracion(e);
  if (est.pestana === "ejecucion") pintarEjecucion(e);
  if (est.pestana === "recursos") pintarRecursos(e);
}

// ---------------------------------------------------------------- 1. topologia
function pintarTopologia(e) {
  const nodos = porNodo(e.workers);
  const config = e.nodos_config || [];
  const nombres = Array.from(new Set(config.map(n => n.hostname).concat(Object.keys(nodos))));
  const master = (config.find(n => (n.roles || []).includes("master")) || {}).hostname;
  // Diagrama: Master al centro, el resto alrededor
  const svg = $("#diagrama"); const cx = 400, cy = 190, r = 150;
  const otros = nombres.filter(n => n !== master);
  let html = "";
  const pos = {};
  otros.forEach((n, i) => {
    const a = -Math.PI / 2 + 2 * Math.PI * i / Math.max(otros.length, 1);
    pos[n] = [cx + r * 1.6 * Math.cos(a), cy + r * Math.sin(a)];
  });
  if (master) pos[master] = [cx, cy];
  const estadoNodo = n => {
    const ws = nodos[n] || [];
    if (!ws.length) return "#8c959f";
    if (ws.some(w => w.estado === "perdido")) return "#cf222e";
    if (ws.some(w => w.estado === "sospechoso")) return "#9a6700";
    return "#1a7f37";
  };
  for (const n of otros) html += `<line x1="${cx}" y1="${cy}" x2="${pos[n][0]}" y2="${pos[n][1]}" stroke="#c4cdd6" stroke-width="2"/>`;
  for (const n of nombres) {
    const [x, y] = pos[n]; const cfg = config.find(c => c.hostname === n) || {};
    const ws = nodos[n] || []; const ip = ws.length ? ws[0].ip : cfg.ip;
    const disp = ws.map(w => w.dispositivo).join("+") || (cfg.dispositivos || []).join("+");
    const ancho = n === master ? 170 : 150;
    html += `<g><rect x="${x - ancho / 2}" y="${y - 30}" width="${ancho}" height="60" rx="8" fill="#fff" stroke="${estadoNodo(n)}" stroke-width="${n === master ? 3 : 2}"/>
      <text x="${x}" y="${y - 12}" text-anchor="middle" font-weight="700">${esc(n)}</text>
      <text x="${x}" y="${y + 4}" text-anchor="middle" fill="#5b6b7c">${esc(ip || "IP sin dato")}</text>
      <text x="${x}" y="${y + 20}" text-anchor="middle" fill="#5b6b7c">${esc((cfg.roles || []).join(", ") || "worker")} · ${esc(disp || "-")}</text></g>`;
  }
  svg.innerHTML = html || `<text x="400" y="190" text-anchor="middle">Sin nodos conectados</text>`;
  const total = e.workers.length; const ok = e.workers.filter(w => w.estado === "conectado").length;
  $("#resumen-cluster").innerHTML = dl([
    ["Rol de este Master", esc(e.rol)], ["Activo desde", hora(e.activo_desde)],
    ["Workers", `${ok} conectados de ${total}`], ["Nodos", String(Object.keys(nodos).length)],
    ["Respaldo", e.respaldo_conectado ? "conectado" : sd("no hay respaldo recibiendo instantáneas")],
    ["Corrida", e.corrida ? esc(e.corrida.estado) : "ninguna"]]);
  $("#tarjetas-nodos").innerHTML = nombres.map(n => {
    const ws = nodos[n] || []; const cfg = config.find(c => c.hostname === n) || {};
    if (!ws.length) return `<div class="tarjeta"><h3>${esc(n)} <span class="etiqueta">sin conectar</span></h3>
      <dl class="datos">${dl([["IP (config)", esc(cfg.ip || "")], ["Roles", esc((cfg.roles || []).join(", "))]])}</dl></div>`;
    const hw = ws[0].hardware || {}; const cpu = hw.cpu || {}; const red = hw.red || {};
    const filasW = ws.map(w => `<span class="etiqueta ${w.estado === "conectado" ? "ok" : w.estado === "perdido" ? "mal" : "aviso"}">${esc(w.dispositivo)}: ${esc(w.estado)} · latido hace ${num(w.edad_latido_s, 1)} s</span>`).join("");
    const gpu = ws.map(w => (w.hardware || {}).gpu).find(g => g && g.disponible);
    return `<div class="tarjeta ${ws.every(w => w.estado === "conectado") ? "conectado" : ws.some(w => w.estado === "perdido") ? "perdido" : "sospechoso"}">
      <h3>${esc(n)} <span class="mono">${esc(red.ip || cfg.ip || "")}</span></h3>${filasW}
      <dl class="datos">${dl([
        ["Rol", esc((cfg.roles || ["worker"]).join(", "))],
        ["CPU", esc(cpu.modelo || "") + (cpu.logicos ? ` · ${cpu.logicos.length} hilos` : "") + (cpu.hibrido ? ` · ${cpu.rendimiento.length} P + ${cpu.eficiencia.length} E` : "")],
        ["RAM", hw.ram_mb ? num(hw.ram_mb / 1024, 1, "GB") : sd()],
        ["GPU", gpu ? esc(gpu.nombre + (gpu.vram_mb ? ` · ${gpu.vram_mb} MB` : "")) : "no"],
        ["Enlace", red.velocidad_mbps ? num(red.velocidad_mbps, 0, "Mb/s") + (red.enlace_lento ? ' <span class="etiqueta aviso">lento</span>' : "") : sd("velocidad del enlace no informada")],
        ["Interfaz", red.interfaz ? esc(red.interfaz) : sd()],
        ["Sistema", esc((hw.sistema || "") + " " + (hw.version_sistema || ""))],
        ["Python / OpenMPI", esc(hw.python || "") + " / " + (hw.openmpi ? esc(hw.openmpi) : sd("sin OpenMPI"))],
        ["NFS", hw.nfs_montado === true ? "montado" : hw.nfs_montado === false ? '<span class="etiqueta mal">no montado</span>' : sd()],
        ["Archivos", esc(Object.keys(ws[0].archivos || {}).join(", ") || "ninguno")]])}</dl></div>`;
  }).join("");
}

// ---------------------------------------------------------------- 2. configuracion
async function cargarArchivos() {
  try {
    const lista = await api("GET", "/api/archivos");
    for (const sel of $$('select[name="archivo"], select[name="archivo_b"]')) {
      const previo = sel.value;
      sel.innerHTML = lista.map(a => `<option value="${esc(a.nombre)}">${esc(a.nombre)} (${bytes(a.tamano).replace(/<[^>]+>/g, "")}, ${a.workers.length} workers)</option>`).join("");
      if (previo) sel.value = previo;
    }
  } catch (err) { $("#error-config").textContent = "No se pudieron leer los archivos: " + err.message; }
}
function pintarConfiguracion(e) {
  const op = $('#form-corrida [name="operacion"]').value;
  $$("#form-corrida [data-op]").forEach(x => { x.style.display = x.dataset.op === op ? "" : "none"; });
  const nodos = porNodo(e.workers);
  const firma = Object.entries(nodos).map(([n, ws]) => n + ws.map(w => w.dispositivo).join()).join("|");
  if (firma === est.firmaNodos) return;
  est.firmaNodos = firma;
  $("#tabla-nodos-config tbody").innerHTML = Object.entries(nodos).map(([n, ws]) => {
    const cpu = ws.find(w => w.dispositivo === "cpu"); const gpu = ws.find(w => w.dispositivo === "gpu");
    const rg = (cpu || ws[0]).rangos || {}; const rgg = (gpu || {}).rangos || {};
    const disp = ws.map(w => `<label class="check"><input type="checkbox" data-disp="${esc(w.dispositivo)}" checked>${esc(w.dispositivo)}</label>`).join("");
    const nucleos = ['<option value="todos">todos</option>'];
    if (rg.hibrido) nucleos.push('<option value="rendimiento">solo rendimiento (P)</option><option value="eficiencia">solo eficiencia (E)</option>');
    nucleos.push('<option value="sin_hermanos">sin hermanos HT</option><option value="manual">manual…</option>');
    return `<tr data-nodo="${esc(n)}"><td><input type="checkbox" data-usar checked></td><td>${esc(n)}</td><td>${disp}</td>
      <td>${cpu ? `<input type="number" data-procesos min="1" max="${rg.procesos_max || 1}" value="${rg.procesos_max || 1}" title="1 a ${rg.procesos_max}">` : "-"}</td>
      <td>${cpu ? `<select data-nucleos>${nucleos.join("")}</select><input data-manual placeholder="0-3,6" style="display:none" title="Núcleos válidos: ${esc((rg.logicos || []).join(","))}">` : "-"}</td>
      <td>${cpu ? `<select data-impl><option value="numpy">numpy</option>${rg.avx2 ? '<option value="simd">SIMD AVX2</option>' : ""}<option value="simd_escalar">C escalar</option></select>` : "-"}</td>
      <td>${gpu ? `<input type="number" data-bloques min="1" max="1024" value="1024">` : "-"}</td>
      <td>${gpu ? `<input type="number" data-lote min="1" max="${rgg.lote_max_mb || 1024}" value="${Math.min(64, rgg.lote_max_mb || 64)}" title="máximo ${rgg.lote_max_mb || "sin dato"} MB">` : "-"}</td></tr>`;
  }).join("") || '<tr><td colspan="8" class="sin-dato">No hay workers conectados</td></tr>';
  $$("#tabla-nodos-config [data-nucleos]").forEach(s => s.addEventListener("change", () => {
    s.nextElementSibling.style.display = s.value === "manual" ? "" : "none";
  }));
  $("#nota-nodos").textContent = "Los rangos de cada campo salen del hardware que detectó cada worker.";
}
function validarCampo(inp) {
  let ok = true;
  if (inp.type === "number" && inp.value !== "") {
    const v = Number(inp.value);
    ok = !Number.isNaN(v) && (inp.min === "" || v >= Number(inp.min)) && (inp.max === "" || v <= Number(inp.max));
  }
  if (inp.dataset.validar === "patrones") {
    const ps = inp.value.split(/[ ,]+/).filter(Boolean).map(p => p.toUpperCase());
    ok = ps.length >= 1 && ps.length <= 10 && ps.every(p => /^[ACGTN]{2,64}$/.test(p));
  }
  if (inp.hasAttribute("data-manual") && inp.style.display !== "none") ok = /^\d+(-\d+)?(,\d+(-\d+)?)*$/.test(inp.value.trim());
  inp.classList.toggle("invalido", !ok);
  return ok;
}
document.addEventListener("input", ev => { if (ev.target.matches("input")) validarCampo(ev.target); });
$('#form-corrida [name="operacion"]').addEventListener("change", () => est.estado && pintarConfiguracion(est.estado));

function leerConfig() {
  const f = $("#form-corrida"); const v = n => f.elements[n].value;
  const errores = [];
  $$("#form-corrida input").forEach(i => { if (!validarCampo(i)) errores.push(i.name || i.dataset.manual !== undefined ? (i.name || "núcleos manuales") : "campo"); });
  const cfg = { operacion: v("operacion"), archivo: v("archivo"), modo: v("modo"), origen_datos: v("origen_datos"),
    estrategia: v("estrategia"), tiempo_objetivo_s: Number(v("tiempo_objetivo_s")), max_tarea_mb: Number(v("max_tarea_mb")),
    verificar_crc: f.elements.verificar_crc.checked, params: {} };
  if (cfg.operacion === "comparacion") { cfg.archivo_b = v("archivo_b"); cfg.params.modo_comparacion = v("modo_comparacion"); }
  if (cfg.operacion === "patrones") {
    cfg.params.patrones = v("patrones").split(/[ ,]+/).filter(Boolean).map(p => p.toUpperCase());
    cfg.params.complemento_inverso = f.elements.complemento.checked;
  }
  if (cfg.operacion === "zonas") { cfg.params.ventana = Number(v("ventana")); cfg.params.paso = Number(v("paso")); }
  cfg.nodos = []; cfg.dispositivos = {}; cfg.motor_nodos = {}; cfg.mpi_nodos = [];
  for (const tr of $$("#tabla-nodos-config tr[data-nodo]")) {
    if (!$("[data-usar]", tr).checked) continue;
    const n = tr.dataset.nodo; cfg.nodos.push(n);
    cfg.dispositivos[n] = $$("[data-disp]", tr).filter(c => c.checked).map(c => c.dataset.disp);
    const m = {};
    if ($("[data-procesos]", tr)) {
      const sel = $("[data-nucleos]", tr).value;
      m.cpu = { procesos: Number($("[data-procesos]", tr).value), impl: $("[data-impl]", tr).value,
        nucleos: sel === "manual" ? $("[data-manual]", tr).value.trim() : sel };
      cfg.mpi_nodos.push({ hostname: n, slots: m.cpu.procesos, impl: m.cpu.impl });
    }
    if ($("[data-lote]", tr)) m.gpu = { bloques: Number($("[data-bloques]", tr).value), lote_mb: Number($("[data-lote]", tr).value) };
    cfg.motor_nodos[n] = m;
  }
  if (!cfg.nodos.length) errores.push("no hay nodos seleccionados");
  if (!cfg.archivo) errores.push("no hay archivo");
  return { cfg, errores, repeticiones: Number(v("repeticiones")) || 1, calentamiento: f.elements.calentamiento.checked };
}
async function esperarCorrida(cid) {
  for (;;) {
    const r = await api("GET", "/api/corridas/" + cid);
    if (r.terminada) return r.resumen;
    await new Promise(ok => setTimeout(ok, 500));
  }
}
$("#form-corrida").addEventListener("submit", async ev => {
  ev.preventDefault();
  const { cfg, errores, repeticiones, calentamiento } = leerConfig();
  const salida = $("#error-config");
  if (errores.length) { salida.textContent = "Revise: " + errores.join(", "); return; }
  salida.textContent = "";
  try {
    if (cfg.modo === "mpi") {
      await api("POST", "/api/mpi", cfg);
      salida.textContent = "Corrida MPI lanzada; el resultado aparecerá en la pestaña Resultados.";
      return;
    }
    $('#pestanas [data-p="ejecucion"]').click();
    const tiempos = [];
    const total = repeticiones + (calentamiento ? 1 : 0);
    let ultimo = null;
    for (let i = 0; i < total; i++) {
      const { corrida_id } = await api("POST", "/api/corridas", cfg);
      ultimo = await esperarCorrida(corrida_id);
      if (!(calentamiento && i === 0) && ultimo.tiempo_s !== null) tiempos.push(ultimo.tiempo_s);
    }
    if (tiempos.length > 1) {
      const o = tiempos.slice().sort((a, b) => a - b); const med = o[Math.floor(o.length / 2)];
      salida.textContent = `Mediana de ${tiempos.length} repeticiones: ${med.toFixed(3)} s (tiempos: ${tiempos.map(t => t.toFixed(3)).join(", ")})`;
    }
  } catch (err) { salida.textContent = err.message; $('#pestanas [data-p="configuracion"]').click(); }
});

// ---------------------------------------------------------------- 3. ejecucion
function pintarEjecucion(e) {
  const c = e.corrida;
  const sel = $("#fallo-worker"); const previo = sel.value;
  sel.innerHTML = e.workers.map(w => `<option>${esc(w.wid)}</option>`).join(""); if (previo) sel.value = previo;
  $("#btn-caida-master").disabled = !e.respaldo_conectado;
  $("#btn-caida-master").title = e.respaldo_conectado ? "" : "No hay Master de respaldo conectado";
  if (!c) { $("#ej-porcentaje").textContent = "sin corrida"; return; }
  $("#ej-id").textContent = c.corrida_id;
  $("#ej-barra").style.width = (100 * (c.progreso || 0)).toFixed(1) + "%";
  $("#ej-porcentaje").textContent = `${c.estado} · ${(100 * (c.progreso || 0)).toFixed(1)} %`;
  $("#ej-datos").innerHTML = dl([["Operación", esc(c.operacion)], ["Archivo", esc(c.config.nombre_a || "")],
    ["Tiempo", c.tiempo_s === null ? sd("aún no empieza") : num(c.tiempo_s, 2, "s")], ["MB/s", num(c.mb_s, 1)],
    ["Procesado", bytes(c.bytes_hechos)], ["Total", bytes(c.largo)], ["Pendiente", bytes(c.pendiente_bytes)],
    ["Reasignaciones", String((c.reasignaciones || []).length)], ["Estrategia", esc(c.config.estrategia || "")],
    ["Datos", esc(c.config.origen_datos || "")], ["Workers", String((c.participantes || []).length)],
    ["CRC", c.config.verificar_crc ? "sí" : "no"]]);
  grafico("graf-mbs", "line", { datasets: [{ label: "MB/s", data: est.serieMbs, borderColor: COLORES.cpu, pointRadius: 0 }] },
    { scales: { x: { type: "linear", title: { display: true, text: "s" } }, y: { beginAtZero: true } }, plugins: { legend: { display: false } } });
  const vel = c.velocidades || {};
  const ids = Object.keys(vel);
  const dispo = id => ((e.workers.find(w => w.wid === id) || {}).dispositivo) || "cpu";
  grafico("graf-workers", "bar", { labels: ids, datasets: [{ label: "MB/s", data: ids.map(i => vel[i]),
    backgroundColor: ids.map(i => COLORES[dispo(i)]) }] }, { indexAxis: "y", plugins: { legend: { display: false } } });
  dibujarGantt(c, e);
  const eventos = (c.reasignaciones || []).map(r => ({ t: r.t, tipo: "reasignacion", texto: `t=${r.t}s · tarea ${r.tarea} de ${r.worker}: ${r.motivo}` }))
    .concat(e.eventos.filter(x => ["caida", "rechazo", "fallo", "error", "aviso"].includes(x.tipo)).slice(-30)
      .map(x => ({ tipo: x.tipo, texto: hora(x.t).replace(/<[^>]+>/g, "") + " · " + x.texto })));
  $("#ej-eventos").innerHTML = eventos.length ? eventos.map(x => `<div class="${x.tipo}">${esc(x.texto)}</div>`).join("") :
    '<div class="sin-dato">Sin reasignaciones ni caídas</div>';
}
function dibujarGantt(c, e) {
  const lienzo = $("#gantt"); const ctx = lienzo.getContext("2d");
  const ancho = lienzo.width = lienzo.clientWidth * devicePixelRatio;
  const filas = (c.participantes || []);
  const altoFila = 22 * devicePixelRatio;
  const alto = lienzo.height = Math.max(60, (filas.length + 1) * altoFila + 20);
  ctx.clearRect(0, 0, ancho, alto);
  const tareas = c.tareas || [];
  const fin = Math.max(c.tiempo_s || 0, ...tareas.map(t => t.t_resultado || 0), 0.001);
  const izq = 130 * devicePixelRatio;
  const x = t => izq + (ancho - izq - 10) * (t / fin);
  ctx.font = `${12 * devicePixelRatio}px system-ui`; ctx.textBaseline = "middle";
  filas.forEach((wid, i) => {
    const y = 10 + i * altoFila;
    ctx.fillStyle = "#1d2733"; ctx.fillText(wid, 4, y + altoFila / 2);
    ctx.fillStyle = "#f0f3f6"; ctx.fillRect(izq, y + 2, ancho - izq - 10, altoFila - 4);
  });
  for (const t of tareas) {
    const i = filas.indexOf(t.worker); if (i < 0 || t.t_asignacion === null) continue;
    const y = 10 + i * altoFila; const a = x(Math.max(0, t.t_asignacion)); const b = x(t.t_resultado ?? c.tiempo_s ?? t.t_asignacion);
    if (t.estado === "DESCARTADA") {
      ctx.setLineDash([4, 3]); ctx.strokeStyle = "#cf222e"; ctx.lineWidth = 2;
      ctx.strokeRect(a, y + 4, Math.max(2, b - a), altoFila - 8); ctx.setLineDash([]);
    } else {
      ctx.fillStyle = COLORES[t.dispositivo] || "#6e7781";
      ctx.globalAlpha = t.estado === "ASIGNADA" ? 0.45 : 1;
      ctx.fillRect(a, y + 4, Math.max(1.5, b - a - 1), altoFila - 8); ctx.globalAlpha = 1;
      if (t.reasignada) { ctx.strokeStyle = "#cf222e"; ctx.lineWidth = 2; ctx.strokeRect(a, y + 4, Math.max(2, b - a - 1), altoFila - 8); }
    }
  }
  ctx.fillStyle = "#5b6b7c"; ctx.fillText(`0 s`, izq, alto - 8); ctx.textAlign = "right"; ctx.fillText(`${fin.toFixed(2)} s`, ancho - 10, alto - 8); ctx.textAlign = "left";
}
$("#btn-detener").addEventListener("click", () => api("POST", "/api/corridas/actual/cancelar").catch(err => alert(err.message)));
$("#btn-fallo").addEventListener("click", () => api("POST", "/api/fallo", { worker: $("#fallo-worker").value, modo: $("#fallo-modo").value }).catch(err => alert(err.message)));
$("#btn-caida-master").addEventListener("click", () => {
  if (confirm("El Master principal se detendrá y el respaldo tomará el control. ¿Continuar?"))
    api("POST", "/api/master/caida").then(r => alert(r.aviso)).catch(err => alert(err.message));
});

// ---------------------------------------------------------------- 4. recursos
function barrasNucleos(valores, rangos) {
  if (!valores || !valores.length) return sd("sin muestras de uso por núcleo");
  const p = new Set(rangos.rendimiento || []); const e = new Set(rangos.eficiencia || []); const logicos = rangos.logicos || valores.map((_, i) => i);
  return `<div class="nucleos">${valores.map((v, i) => {
    const id = logicos[i] ?? i; const color = p.has(id) ? "var(--nucleo-p)" : e.has(id) ? "var(--nucleo-e)" : "#6e7781";
    return `<div title="núcleo ${id}: ${v.toFixed(0)} %"><span style="height:${Math.min(100, v)}%;background:${color}"></span></div>`;
  }).join("")}</div>`;
}
function pintarRecursos(e) {
  $("#tarjetas-recursos").innerHTML = e.workers.map(w => {
    const m = w.metricas || {}; const mot = m.motivos_energia || {};
    const pot = (k, motivo) => m[k] === null || m[k] === undefined ? sd(motivo) : num(m[k], 1, "W");
    const ene = (k, motivo) => m[k] === null || m[k] === undefined ? sd(motivo) : num(m[k], 1, "J");
    return `<div class="tarjeta ${esc(w.estado)}"><h3>${esc(w.wid)} <span class="etiqueta">${esc(w.estado)}</span></h3>
      ${barrasNucleos(m.cpu_nucleos, w.rangos || {})}
      <dl class="datos">${dl([["CPU", num(m.cpu_pct, 0, "%")], ["Frecuencia", num(m.frecuencia_mhz, 0, "MHz")],
        ["Temperatura CPU", m.temp_cpu_c === null || m.temp_cpu_c === undefined ? sd("sin sensor legible") : num(m.temp_cpu_c, 0, "°C")],
        ["RAM", num(m.ram_pct, 0, "%") + (m.ram_usada_mb ? ` (${num(m.ram_usada_mb / 1024, 1, "GB")})` : "")],
        ["Memoria del worker", num(m.rss_worker_mb, 0, "MB")],
        ["Red", m.red_tx_mb_s === null || m.red_tx_mb_s === undefined ? sd("una sola muestra") : `↑ ${num(m.red_tx_mb_s, 2)} · ↓ ${num(m.red_rx_mb_s, 2)} MB/s`],
        ["GPU", m.gpu_uso_pct === undefined || m.gpu_uso_pct === null ? sd(mot.gpu) : `${num(m.gpu_uso_pct, 0, "%")} · ${num(m.gpu_mem_mb, 0, "MB")} · ${num(m.gpu_temp_c, 0, "°C")}`],
        ["Potencia CPU / GPU / ANE", `${pot("cpu_w", mot.cpu)} / ${pot("gpu_w", mot.gpu)} / ${pot("ane_w", mot.mac)}`],
        ["Energía acumulada", `CPU ${ene("cpu_j", mot.cpu)} · GPU ${ene("gpu_j", mot.gpu)} · ANE ${ene("ane_j", mot.mac)}`]])}</dl></div>`;
  }).join("") || '<p class="sin-dato">No hay workers conectados</p>';
  const series = Object.entries(est.cpuHist).map(([wid, h], i) => ({ label: wid, data: h.map(p => ({ x: p.t - (h[h.length - 1] || p).t, y: p.y })),
    borderColor: `hsl(${(i * 67) % 360} 65% 45%)`, pointRadius: 0 }));
  grafico("graf-cpu", "line", { datasets: series }, { scales: { x: { type: "linear", title: { display: true, text: "s (0 = ahora)" } },
    y: { min: 0, max: 100, title: { display: true, text: "%" } } } });
}

// ---------------------------------------------------------------- 5. resultados
async function cargarResultados() {
  try {
    const filas = await api("GET", "/api/resultados");
    $("#lista-resultados").innerHTML = filas.map(f => {
      const id = f.corrida_id || (f.carpeta || "").split("/").pop();
      return `<div data-id="${esc(id)}"><b>${esc(f.operacion || "")}</b> ${f.modo === "mpi" ? `MPI (${f.ranks})` : ""} · ${esc(f.estado || "")}
        · ${f.tiempo_s === null || f.tiempo_s === undefined ? sd() : num(f.tiempo_s, 3, "s")}
        ${f.valido === true ? '<span class="etiqueta ok">válida</span>' : f.valido === false ? '<span class="etiqueta mal">no válida</span>' : ""}
        <div class="mono">${esc(id)}</div></div>`;
    }).join("") || '<p class="sin-dato">Todavía no hay corridas</p>';
    $$("#lista-resultados [data-id]").forEach(d => d.addEventListener("click", () => {
      $$("#lista-resultados [data-id]").forEach(x => x.classList.toggle("sel", x === d)); verResultado(d.dataset.id);
    }));
  } catch (err) { $("#lista-resultados").textContent = err.message; }
}
function tabla(cab, filas) {
  return `<table class="tabla"><thead><tr>${cab.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${
    filas.map(f => `<tr>${f.map(v => `<td${typeof v === "number" ? ' class="num"' : ""}>${typeof v === "number" ? num(v, 3) : v ?? sd()}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
async function verResultado(id) {
  est.resultadoSel = id;
  const h = await api("GET", "/api/resultados/" + encodeURIComponent(id));
  $("#res-id").textContent = id;
  const v = h.validacion || {}; const r = h.resultado || {};
  const nucleos = (h.por_worker || []).reduce((s, p) => s + (((p.motor || {}).procesos) || 1), 0) || h.ranks || null;
  let html = `<dl class="datos en-linea">${dl([["Operación", esc(h.operacion || "")], ["Modo", esc(h.modo || "dinámico")],
    ["Tiempo", num(h.tiempo_s, 3, "s")], ["Preparación (aparte)", num(h.preparacion_s, 3, "s")], ["MB/s", num(h.mb_s, 1)],
    ["Speedup", h.speedup ? num(h.speedup, 2) : sd("no hay referencia secuencial")],
    ["Eficiencia", h.speedup && nucleos ? num(h.speedup / nucleos, 3) : sd()], ["Núcleos", nucleos ? String(nucleos) : sd()],
    ["Validación", v.valido ? '<span class="etiqueta ok">válida</span>' : '<span class="etiqueta mal">revisar</span>'],
    ["Referencia", v.coincide_referencia === true ? "coincide" : v.coincide_referencia === false ? '<span class="etiqueta mal">discrepa</span>' : sd("no hay resultado de referencia")],
    ["CRC", v.crc_verificado ? "verificado" : "no"], ["Reasignaciones", String((h.reasignaciones || []).length)]])}</dl>`;
  if (v.discrepancias && v.discrepancias.length) html += `<p class="error">${v.discrepancias.map(esc).join("<br>")}</p>`;
  const arq = h.energia_por_arquitectura || {};
  if (Object.keys(arq).length) html += "<h3>Energía por arquitectura</h3>" + tabla(["Dispositivo", "Energía (J)", "MB", "MB/J"],
    Object.entries(arq).map(([k, a]) => [esc(k), a.energia_j, a.bytes / 1048576, a.mb_por_j]));
  if (h.por_worker) html += "<h3>Reparto por worker</h3>" + tabla(["Worker", "Disp.", "MB", "Tareas", "MB/s final", "Ocioso final (s)"],
    h.por_worker.map(p => [esc(p.worker), esc(p.dispositivo), p.bytes / 1048576, p.tareas, p.mb_s_final, p.ocioso_final_s]));
  if (h.por_rank) html += "<h3>Reparto por rank (MPI)</h3>" + tabla(["Rank", "Host", "MB", "t local (s)", "Afinidad"],
    h.por_rank.map(p => [p.rank, esc(p.host), p.bytes / 1048576, p.t_local_s, esc((p.afinidad || []).join(","))]));
  html += resultadoOperacion(h.operacion, r);
  $("#res-detalle").innerHTML = html;
  const filas = h.por_worker || h.por_rank || [];
  grafico("graf-res", "bar", { labels: filas.map(p => p.worker || ("rank " + p.rank)), datasets: [{ label: "MB procesados",
    data: filas.map(p => p.bytes / 1048576), backgroundColor: filas.map(p => COLORES[p.dispositivo] || COLORES.cpu) }] },
    { plugins: { legend: { display: false } }, maintainAspectRatio: false });
}
function loc(x) { return x.nombre !== undefined ? `${esc(x.nombre)} fila ${x.fila}, col ${x.columna}` : sd(); }
function resultadoOperacion(op, r) {
  if (!r || r.error) return r && r.error ? `<p class="error">${esc(r.error)}</p>` : "";
  if (op === "conteo") return "<h3>Conteo</h3>" + tabla(["A", "C", "G", "T", "N", "IUPAC", "Inválidos", "Total"],
    [[r.A, r.C, r.G, r.T, r.N, r.iupac_total, r.invalidos_total, r.total]]) +
    (r.primeros_invalidos && r.primeros_invalidos.length ? "<h3>Primeros inválidos</h3>" + tabla(["Posición", "Byte", "Registro, fila y columna"],
      r.primeros_invalidos.slice(0, 20).map(x => [x.posicion_global, esc(x.byte), loc(x)])) : "");
  if (op === "patrones") return "<h3>Patrones</h3>" + tabla(["Patrón", "Total", "+", "−", "Palíndromo", "Primeras posiciones"],
    (r.patrones || []).map(p => [esc(p.patron), p.total, p["+"], p["-"], p.palindromo ? "sí" : "no",
      (p.primeras || []).slice(0, 3).map(x => `${x.posicion_global} (${x.hebra}) ${loc(x)}`).join("<br>")]));
  if (op === "comparacion") return "<h3>Comparación</h3>" + tabla(["Comparadas", "Reales", "Solo caso", "Con N", "Total"],
    [[r.comparadas, r.reales, r.solo_caso, r.con_n, r.total]]) +
    (r.parejas ? `<p class="nota">${r.num_parejas} parejas (${r.por_cromosoma} por cromosoma, ${r.por_longitud} por longitud); sin pareja: ${(r.sin_pareja_a || []).length} en A, ${(r.sin_pareja_b || []).length} en B</p>` +
      tabla(["Pareja", "Criterio", "Comparadas", "Diferencias"], r.parejas.slice(0, 30).map(p => [esc(p.nombre), esc(p.criterio), p.comparadas, p.total])) : "") +
    ((r.primeras || []).length ? "<h3>Primeras diferencias</h3>" + tabla(["Pos. A", "A", "B", "Categoría", "Registro, fila y columna"],
      r.primeras.slice(0, 20).map(d => [d.posicion_a, esc(d.byte_a), esc(d.byte_b), esc(d.categoria), loc(d)])) : "");
  if (op === "zonas") return "<h3>Zonas de interés</h3>" + tabla(["Evaluadas", "Positivas", "No evaluables"], [[r.evaluadas, r.positivas, r.no_evaluables]]) +
    tabla(["Registro", "Evaluadas", "Positivas", "Densidad"], (r.por_registro || []).slice(0, 30).map(x => [esc(x.nombre), x.evaluadas, x.positivas, x.densidad])) +
    ((r.primeras || []).length ? "<h3>Primeras zonas</h3>" + tabla(["Registro", "Inicio", "GC", "Obs/esp", "Prob."],
      r.primeras.slice(0, 15).map(z => [esc(z.nombre || z.registro), z.inicio ?? z.inicio_global, z.gc, z.obs_esp, z.probabilidad])) : "");
  return "";
}
$$("[data-exportar]").forEach(b => b.addEventListener("click", () => {
  if (est.resultadoSel) location.href = `/api/resultados/${encodeURIComponent(est.resultadoSel)}/exportar?formato=${b.dataset.exportar}`;
}));
$("#btn-png").addEventListener("click", () => {
  for (const id of ["graf-res", "gantt"]) {
    const c = document.getElementById(id); if (!c) continue;
    const a = document.createElement("a"); a.download = `${est.resultadoSel || "corrida"}_${id}.png`; a.href = c.toDataURL("image/png"); a.click();
  }
});

// ---------------------------------------------------------------- 6. escalabilidad
async function cargarEscalabilidad() {
  await cargarArchivos();
  const e = await api("GET", "/api/escalabilidad");
  $("#escala-estado").textContent = e.corriendo ? "Serie en curso…" : e.error ? "Error: " + e.error : e.resultado ? "Última serie: " + (e.resultado.carpeta || "") : "Sin series todavía";
  if (e.corriendo) setTimeout(() => est.pestana === "escalabilidad" && cargarEscalabilidad(), 2000);
  if (!e.resultado) return;
  const filas = (e.resultado.filas || []).filter(f => f.tiempo_s !== undefined && f.tiempo_s !== "" && f.tiempo_s !== null);
  const series = {};
  for (const f of filas) (series[f.serie] = series[f.serie] || []).push(f);
  const ds = clave => Object.entries(series).filter(([s]) => s !== "1_referencia").map(([s, fs], i) => ({ label: s,
    data: fs.map(f => ({ x: Number(f.nucleos), y: Number(f[clave]) })), borderColor: `hsl(${i * 90} 65% 45%)`, showLine: true }));
  const opc = t => ({ scales: { x: { type: "linear", title: { display: true, text: "núcleos" } }, y: { beginAtZero: true, title: { display: true, text: t } } } });
  grafico("graf-esc-t", "scatter", { datasets: ds("tiempo_s") }, opc("tiempo (s)"));
  const maxN = Math.max(1, ...filas.map(f => Number(f.nucleos)));
  const ideal = { label: "ideal", data: [{ x: 1, y: 1 }, { x: maxN, y: maxN }], borderColor: "#8c959f", borderDash: [4, 4], showLine: true, pointRadius: 0 };
  grafico("graf-esc-s", "scatter", { datasets: ds("speedup").concat([ideal]) }, opc("speedup"));
  grafico("graf-esc-e", "scatter", { datasets: ds("eficiencia") }, opc("eficiencia"));
  const am = e.resultado.amdahl || {};
  $("#escala-amdahl").innerHTML = Object.keys(am).length ? tabla(["Serie", "Fracción secuencial", "Speedup máximo", "R²"],
    Object.entries(am).map(([s, a]) => [esc(s), a.fraccion_secuencial, a.speedup_maximo, a.r2])) : "";
  $("#escala-tabla").innerHTML = tabla(["Serie", "Punto", "Núcleos", "Tiempo (s)", "Speedup", "Eficiencia", "Válida"],
    (e.resultado.filas || []).map(f => [esc(f.serie), esc(f.punto), f.nucleos, f.tiempo_s === "" ? null : Number(f.tiempo_s),
      f.speedup === "" ? null : Number(f.speedup), f.eficiencia === "" ? null : Number(f.eficiencia), esc(String(f.valido ?? f.error ?? ""))]));
}
$("#form-escala").addEventListener("submit", async ev => {
  ev.preventDefault(); const f = ev.target;
  const cfg = { operacion: f.elements.operacion.value, archivo: f.elements.archivo.value,
    nucleos: f.elements.nucleos.value.split(",").map(Number).filter(n => n > 0), repeticiones: Number(f.elements.repeticiones.value),
    calentamiento: f.elements.calentamiento.checked, solo_local: f.elements.solo_local.checked, dinamico: f.elements.dinamico.checked };
  try { await api("POST", "/api/escalabilidad", cfg); cargarEscalabilidad(); } catch (err) { $("#escala-estado").textContent = err.message; }
});

// ---------------------------------------------------------------- 7. evidencias
async function cargarEvidencias() {
  const filas = await api("GET", "/api/evidencias");
  $("#tabla-evidencias").innerHTML = tabla(["Criterio", "Evidencia", "Estado", "Detalle"], filas.map(f => [esc(f.criterio), esc(f.evidencia),
    f.estado === "con dato" ? '<span class="etiqueta ok">con dato</span>' : '<span class="etiqueta aviso">sin dato</span>',
    f.valor === null ? `<span class="sin-dato">${esc(f.motivo || "")}</span>` : `<div class="mono detalle-json">${esc(JSON.stringify(f.valor)).slice(0, 700)}</div>`]));
}
$("#btn-evidencias").addEventListener("click", cargarEvidencias);
$("#btn-evidencias-md").addEventListener("click", () => api("POST", "/api/evidencias/exportar").then(r => alert("Escrito " + r.ruta)).catch(err => alert(err.message)));

// ---------------------------------------------------------------- 8. registro
async function cargarRegistro() {
  if (est.pestana !== "registro") return;
  try { const t = await fetch("/api/registro").then(r => r.text()); const p = $("#log"); p.textContent = t; p.scrollTop = p.scrollHeight; } catch { /* reintenta */ }
  setTimeout(cargarRegistro, 3000);
}

// ---------------------------------------------------------------- inicio
cargarArchivos();
setInterval(cargarArchivos, 15000);
conectar();
