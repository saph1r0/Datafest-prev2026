"""
PASO 5 — Simulador para la sustentación: un solo archivo HTML, sin servidor y sin internet.

Por qué HTML estático y no Streamlit: abre en cualquier laptop con doble clic, no depende de que el
entorno de Python o la red de la sala funcionen, y se puede enviar como un archivo. Streamlit es
una buena alternativa si tienen tiempo y la sala lo permite.

El jurado puede mover la capacidad comercial, filtrar por región y comparar el orden del modelo
contra los leads actuales. Responde en vivo: "¿y si tuviéramos la mitad de ejecutivos?".
"""
import json

import pandas as pd

from utils import cargar_config

PLANTILLA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simulador de priorización</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--soft:#6b6b66;--line:#e2e1dc;--acc:#0b5cad;--acc2:#e07a1f;--ok:#2e7d4f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--soft:#a3a29c;--line:#34332f;--acc:#5aa2ec;--acc2:#f0a05a;--ok:#5cc08a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--soft);margin:0 0 20px}
.ctrl,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.ctrl{display:grid;grid-template-columns:2fr 1fr 1fr;gap:16px;margin-bottom:16px}
label{display:block;font-size:12px;color:var(--soft);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px}
input[type=range]{width:100%}select,input[type=number]{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);font:inherit}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.kpi .v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}.kpi .c{font-size:13px;color:var(--soft)}
.up{color:var(--ok);font-weight:600}
.grid{display:grid;grid-template-columns:3fr 2fr;gap:16px;margin-bottom:16px}
h2{font-size:15px;margin:0 0 10px}
.bar{display:grid;grid-template-columns:150px 1fr 54px;gap:8px;align-items:center;font-size:13px;margin:5px 0}
.bar i{display:block;height:12px;border-radius:3px;background:var(--acc)}.bar span:last-child{text-align:right;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--soft);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.leg{display:flex;gap:16px;font-size:12px;color:var(--soft);margin-top:6px;flex-wrap:wrap}.leg b{display:inline-block;width:12px;height:3px;margin-right:6px;vertical-align:middle}
@media (max-width:760px){.ctrl,.grid{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}.bar{grid-template-columns:110px 1fr 48px}}
</style></head><body><main>
<h1>¿A qué empresas llamar este trimestre?</h1>
<p class="sub" id="frase"></p>
<section class="ctrl">
  <div><label for="cap">Capacidad comercial: <span id="capv"></span> contactos</label><input id="cap" type="range" min="10" step="10"></div>
  <div><label for="reg" id="lblB">Grupo</label><select id="reg"></select></div>
  <div><label for="ord">Orden de la lista</label><select id="ord">
    <option value="ve">Modelo: valor esperado</option><option value="p">Modelo: probabilidad</option><option value="lead" id="optLead">Leads actuales</option></select></div>
</section>
<section class="kpis">
  <div class="card kpi"><div class="c">Conversiones esperadas</div><div class="v" id="k1"></div><div class="c" id="k1c"></div></div>
  <div class="card kpi"><div class="c">Monto esperado de factoring</div><div class="v" id="k2"></div><div class="c" id="k2c"></div></div>
  <div class="card kpi"><div class="c">Ingreso esperado para el banco</div><div class="v" id="k3"></div><div class="c" id="k3c"></div></div>
  <div class="card kpi"><div class="c">Del ingreso total posible</div><div class="v" id="k4"></div><div class="c">capturado con esta capacidad</div></div>
</section>
<section class="grid">
  <div class="card"><h2>Ingreso esperado acumulado según a quién llamas primero</h2><p class="sub" style="margin:-6px 0 8px;font-size:12px">Estimado por el modelo. La prueba con resultados reales (validación out-of-time) está en resumen_negocio.md.</p><svg id="g" viewBox="0 0 600 300" width="100%" role="img" aria-label="Curva de ingreso acumulado"></svg>
  <div class="leg"><span><b style="background:var(--acc)"></b>Modelo: valor esperado</span><span class="legLead"><b style="background:var(--acc2)"></b>Leads actuales</span><span><b style="background:var(--soft)"></b>Azar</span></div></div>
  <div class="card"><h2 id="lblA2">Grupos en la lista seleccionada</h2><div id="rub"></div></div>
</section>
<section class="card"><h2>Primeras 15 empresas y por qué</h2><table><thead><tr><th>#</th><th>ID</th><th id="thA">A</th><th id="thB">B</th><th class="n">Prob.</th><th class="n">Monto est.</th><th>Razones</th></tr></thead><tbody id="tb"></tbody></table></section>
</main>
<script>
const D = __DATOS__;
const mon = D.hayMonto ? "S/ " : "";
const fmt = n => n >= 1e6 ? mon + (n/1e6).toFixed(2) + " M" : n >= 1e3 ? mon + (n/1e3).toFixed(0) + " mil" : mon + n.toFixed(D.hayMonto ? 0 : 1);
const $ = id => document.getElementById(id);
const regiones = ["Todas", ...D.regiones];
$("lblB").textContent = D.nombreB; $("lblA2").textContent = D.nombreA + " en la lista seleccionada";
$("thA").textContent = D.nombreA; $("thB").textContent = D.nombreB;
if (!D.hayLead) { $("optLead").remove(); document.querySelectorAll(".legLead").forEach(e => e.remove()); }
$("reg").innerHTML = regiones.map(r => `<option>${r}</option>`).join("");
$("frase").textContent = D.frase;
const filas = D.filas.map(f => ({id:f[0], rubro:D.rubros[f[1]], region:D.regiones[f[2]], lead:f[3], p:f[4], m:f[5], raz:f[6], azar:f[7]}));
filas.forEach(f => { f.me = f.p * f.m; f.ing = f.me * D.margen; });
const cap = $("cap");
function universo(){ const r = $("reg").value; return r === "Todas" ? filas : filas.filter(f => f.region === r); }
function ordenar(u, o){
  const c = u.slice();
  if (o === "ve") c.sort((a,b) => b.ing - a.ing);
  else if (o === "p") c.sort((a,b) => b.p - a.p);
  else if (o === "lead") c.sort((a,b) => (b.lead - a.lead) || (a.azar - b.azar));
  else c.sort((a,b) => a.azar - b.azar);
  return c;
}
const suma = (a, k) => a.reduce((s, f) => s + f[k], 0);
function curva(lista, n){ const pts = []; let acc = 0; const paso = Math.max(1, Math.floor(lista.length / 120));
  for (let i = 0; i < lista.length; i++){ acc += lista[i].ing; if (i % paso === 0 || i === lista.length - 1) pts.push([i + 1, acc]); } return pts; }
function dibujar(){
  const u = universo(); cap.max = u.length; if (+cap.value > u.length) cap.value = u.length;
  const n = +cap.value; $("capv").textContent = n.toLocaleString("es-PE") + " de " + u.length.toLocaleString("es-PE");
  const o = $("ord").value, lista = ordenar(u, o), sel = lista.slice(0, n), leads = ordenar(u, "lead").slice(0, n);
  const tot = suma(u, "ing");
  $("k1").textContent = suma(sel, "p").toFixed(1); $("k2").textContent = fmt(suma(sel, "me")); $("k3").textContent = fmt(suma(sel, "ing"));
  $("k4").textContent = (100 * suma(sel, "ing") / tot).toFixed(0) + "%";
  const ref = D.hayLead ? leads : ordenar(u, "azar").slice(0, n), nomRef = D.hayLead ? "leads actuales" : "llamar al azar";
  const comp = (a, b) => b > 0 ? `<span class="up">${(a / b).toFixed(1)}x</span> vs ${nomRef}` : "";
  $("k1c").innerHTML = o === "lead" ? "práctica actual" : comp(suma(sel, "p"), suma(ref, "p"));
  $("k2c").innerHTML = o === "lead" ? "práctica actual" : comp(suma(sel, "me"), suma(ref, "me"));
  $("k3c").innerHTML = o === "lead" ? "práctica actual" : comp(suma(sel, "ing"), suma(ref, "ing"));
  // curva
  const W = 600, H = 300, L = 56, B = 34, R = 12, T = 12, N = u.length;
  const X = i => L + (W - L - R) * i / N, Y = v => H - B - (H - B - T) * v / tot;
  const linea = (pts, color, ancho) => `<polyline fill="none" stroke="${color}" stroke-width="${ancho}" points="${[[0,0], ...pts].map(p => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join(" ")}"/>`;
  let s = "";
  for (let k = 0; k <= 4; k++){ const v = tot * k / 4; s += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end" font-size="11" fill="var(--soft)">${k * 25}%</text>`; }
  s += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle" font-size="11" fill="var(--soft)">empresas contactadas</text>`;
  s += linea(curva(ordenar(u, "azar")), "var(--soft)", 1.5) + (D.hayLead ? linea(curva(ordenar(u, "lead")), "var(--acc2)", 2) : "") + linea(curva(ordenar(u, "ve")), "var(--acc)", 2.5);
  s += `<line x1="${X(n)}" x2="${X(n)}" y1="${T}" y2="${H - B}" stroke="var(--ink)" stroke-dasharray="4 4"/><text x="${X(n) + 6}" y="${T + 12}" font-size="11" fill="var(--ink)">capacidad</text>`;
  $("g").innerHTML = s;
  // rubros
  const cnt = {}; sel.forEach(f => cnt[f.rubro] = (cnt[f.rubro] || 0) + f.ing);
  const top = Object.entries(cnt).sort((a,b) => b[1] - a[1]); const mx = top.length ? top[0][1] : 1; const st = suma(sel, "ing") || 1;
  $("rub").innerHTML = top.map(([r, v]) => `<div class="bar"><span>${r}</span><i style="width:${(100 * v / mx).toFixed(1)}%"></i><span>${(100 * v / st).toFixed(0)}%</span></div>`).join("");
  $("tb").innerHTML = sel.slice(0, 15).map((f, i) => `<tr><td>${i + 1}</td><td>${f.id}</td><td>${f.rubro}</td><td>${f.region}</td><td class="n">${(100 * f.p).toFixed(1)}%</td><td class="n">${fmt(f.m)}</td><td>${f.raz}</td></tr>`).join("");
}
cap.max = filas.length;  // primero el máximo: un range nace con max=100 y recortaría el valor
cap.value = Math.min(D.capacidad, filas.length);
["input", "change"].forEach(ev => { cap.addEventListener(ev, dibujar); $("reg").addEventListener(ev, dibujar); $("ord").addEventListener(ev, dibujar); });
dibujar();
</script></body></html>"""


def main():
    cfg = cargar_config()
    sal = cfg["rutas"]["salidas"]
    lista = pd.read_csv(sal / "lista_priorizada.csv").drop_duplicates("id")
    resumen = json.loads((sal / "resumen_negocio.json").read_text(encoding="utf-8"))
    pc = [c for c in resumen.get("perfil_columnas") or [] if c in lista.columns]
    ga = pc[0] if pc else None
    gb = pc[1] if len(pc) > 1 else ga
    lead = resumen.get("columna_lead")
    va = lista[ga].astype(str) if ga else pd.Series("TODOS", index=lista.index)
    vb = lista[gb].astype(str) if gb else pd.Series("TODOS", index=lista.index)
    rubros, regiones = sorted(va.unique()), sorted(vb.unique())
    ia, ib = {v: i for i, v in enumerate(rubros)}, {v: i for i, v in enumerate(regiones)}
    leads = lista[lead].astype(bool).astype(int) if lead else pd.Series(0, index=lista.index)
    azar = pd.Series(range(len(lista))).sample(frac=1, random_state=3).to_numpy()
    filas = [[str(i), ia[a], ib[b], int(l), round(float(p), 5), round(float(m), 0),
              str(r) if pd.notna(r) else "", int(z)]
             for i, a, b, l, p, m, r, z in zip(lista.id, va, vb, leads, lista.p, lista.monto_estimado,
                                               lista.get("razones", pd.Series("", index=lista.index)), azar)]
    hay_monto = bool(resumen.get("hay_monto", True))
    datos = {"filas": filas, "rubros": rubros, "regiones": regiones,
             "margen": cfg["negocio"]["margen"] if hay_monto else 1.0, "hayMonto": hay_monto,
             "hayLead": bool(lead), "nombreA": (ga or "grupo").capitalize(), "nombreB": (gb or "grupo").capitalize(),
             "capacidad": cfg["negocio"]["capacidad_contactos"], "frase": resumen["frase"]}
    html = PLANTILLA.replace("__DATOS__", json.dumps(datos, ensure_ascii=False))
    (sal / "simulador.html").write_text(html, encoding="utf-8")
    print(f"  simulador: {sal / 'simulador.html'} ({len(html) / 1e6:.1f} MB) — ábranlo con doble clic")


if __name__ == "__main__":
    main()
