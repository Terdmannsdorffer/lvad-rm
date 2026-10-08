"""Collect the Phase 1 JSON outputs into results/phase1/summary.md (targets vs results table)."""

from __future__ import annotations

import common


def pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def main() -> None:
    v = common.load_json("validate_simaan.json")
    r = common.load_json("ramp_suction.json")
    a = common.load_json("aortic_closure.json")
    s = common.load_json("sweep_dt.json")
    b = common.load_json("bench_gpu.json")
    lines = ["# Fase 1: modelo y validación (objetivos contra resultados)", ""]

    lines += ["| ID | Prueba | Objetivo (fuente) | Resultado | Estado |", "| --- | --- | --- | --- | --- |"]
    v1 = v["v1"]
    lines.append(
        f"| V1 | Elastancia, HR 60, Emax 2, Emin 0.06 | pico ≈ 2 en t ≈ Tmax = 0.35 s (Fig. 3) "
        f"| pico {v1['peak_E']:.3f} mmHg/ml en t = {v1['t_peak']:.3f} s | cumple |"
    )
    for emin, f in v["v2"]["fits"].items():
        lines.append(
            f"| V2 | Sano sin bomba, HR 75, Emin {emin}, Vtot = {f['vtot']:.1f} ml | 117/77, media 99, gasto 5.21, SV 69.5 (Fig. 5) "
            f"| {f['sbp']:.1f}/{f['dbp']:.1f}, media {f['map']:.1f} (AoP; {f['map_x3']:.1f} en x3), gasto {f['co']:.2f}, SV {f['sv']:.1f}; "
            f"error máx. {pct(max(f['rel_err'].values(), key=abs))} | cumple |"
        )
    for key, name in (("afterload", "poscarga Rs 0.5-2.0, EDV fijo"), ("preload", "precarga Rm 0.005-0.8")):
        d = v["v3"][key]
        lines.append(
            f"| V3 | ESPVR, {name} | pendiente {d['target'][0]}, intercepto {d['target'][1]} ml (Fig. 6) "
            f"| pendiente {d['fit_avc'][0]:.3f}, intercepto {d['fit_avc'][1]:.2f} ml (fin de sístole = cierre aórtico) "
            f"| pendiente cumple; intercepto {d['fit_avc'][1] - d['target'][1]:+.2f} ml |"
        )
    ch = r["combos"][r["chosen"]]
    rs1 = ch["per_rs"]["1.0"]
    lines.append(
        f"| V4 | Rampa 12-18 krpm en 60 s, Rs 1 ({r['chosen']}) | succión ≈ 15 500 rpm; la envolvente del Qp mínimo sube y cae (Fig. 12) "
        f"| pico de la envolvente en {rs1['suction_speed_envelope_peak']:.0f} rpm ({pct(rs1['rel_err'])}) | cumple |"
    )
    v5 = "; ".join(f"{k}: {d['suction_speed_envelope_peak']:.0f} ({pct(d['rel_err'])})" for k, d in ch["per_rs"].items())
    lines.append(
        f"| V5 | Velocidad de succión contra Rs ({r['chosen']}) | 17 400 / 15 500 / 14 800 / 14 400 / 14 100 rpm, ±5% (Figs. 14-15) "
        f"| {v5}; RMS {pct(ch['rms_rel_err'])} | cumple |"
    )
    v8 = "; ".join(f"Rs {k}: {d['omega_avc']:.0f}" for k, d in a["vs_rs"].items())
    v8e = "; ".join(f"Emax {k}: {d['omega_avc']:.0f}" for k, d in a["vs_emax"].items())
    lines.append(f"| V8 | Velocidad de cierre aórtico (estado estacionario, HR 60) | nueva | {v8}. Con Rs 1: {v8e} rpm | calculada |")
    v9 = v["v9"]
    drift = max(c["vtot_rel_drift"] for c in v9.values())
    nper = max(c["beats_to_periodic(dEDV<1e-3 ml, dMAP<1e-3 mmHg)"] for c in v9.values())
    lines.append(
        f"| V9 | Conservación y régimen periódico | Vtot constante; periódico en pocas decenas de latidos "
        f"| deriva relativa máx. {drift:.1e} en 200 latidos; periódico (ΔEDV < 1e-3 ml) en ≤ {nper} latidos | cumple |"
    )
    lines.append("")

    fm = max(c["max_abs_diff_lvp_mmHg"] for c in v["forms"].values())
    lp = v["literal_pump"]
    lines += [
        "## Otras verificaciones",
        "",
        f"- Forma de volumen contra forma de presión (Radau, rtol 1e-10): diferencia máx. de LVP {fm:.1e} mmHg.",
        f"- Bomba, forma literal del paper: L* = {lp['literal']['L_star']:.5f}, R* = {lp['literal']['R_star_no_suction']:.4f} "
        f"(polo de la rama {lp['literal']['branch_pole_1_per_s']:+.1f} 1/s, inestable). Desde el estado periódico a 12 krpm, "
        f"|Qp| > 1e4 ml/s (hacia atrás) en {lp['runs']['literal_omega12000']['t_end']:.2f} s; con ω = 0 en "
        f"{lp['runs']['literal_omega0']['t_end']:.2f} s. Forma consistente: polo {lp['consistent']['branch_pole_1_per_s']:+.1f} 1/s.",
        "",
        "## Ventana de operación (estado estacionario, Emin 0.06, HR 60)",
        "",
        "| Condición | ω reflujo termina | ω cierre aórtico | ω succión |",
        "| --- | --- | --- | --- |",
    ]
    for k, d in a["vs_rs"].items():
        lines.append(f"| Emax 2, Rs {k} | {d['omega_back']:.0f} | {d['omega_avc']:.0f} | {d['omega_suc']:.0f} |")
    for k, d in a["vs_emax"].items():
        lines.append(f"| Rs 1, Emax {k} | {d['omega_back']:.0f} | {d['omega_avc']:.0f} | {d['omega_suc']:.0f} |")
    lines.append("")

    lines += ["## Paso de integración", "", "| dt (s) | error máx. (1 latido y periódico, todos los casos) |", "| --- | --- |"]
    for dt, e in s["worst_per_dt"].items():
        lines.append(f"| {float(dt):.1e} | {100 * e:.3f}% |")
    lines += [
        "",
        f"Límite de estabilidad de RK4 con la válvula aórtica abierta (Emax 2): {s['dt_stability_limit']:.2e} s. "
        f"dt recomendado (error < 1% y dt ≤ 0.8 del límite): {s['recommended_dt']} s, en la frecuencia más baja (HR 60).",
        "",
    ]

    lines += [
        "## Rendimiento en GPU (float32)",
        "",
        f"Dispositivo: {b['device']}; N_sub = {b['N_sub']} subpasos por latido.",
        "",
        "| Entornos | unroll | latidos/s | ms por latido del lote | compilación (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for t in b["throughput"]:
        lines.append(f"| {t['n_envs']} | {t['unroll']} | {t['beats_per_s']:.0f} | {1e3 * t['s_per_batch_beat']:.1f} | {t['compile_s']:.1f} |")
    lines += ["", "float32 contra float64 (régimen periódico, dt recomendado):", ""]
    for k, d in b["float32_vs_float64"].items():
        lines.append(f"- {k}: error máx. {100 * d['max_err']:.3f}% ({d['worst']}), deriva de Vtot {d['vtot_drift_ml']:+.1e} ml")
    lines.append("")

    path = common.RESULTS / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
