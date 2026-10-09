"""Collect the Phase 2 JSON outputs into results/phase2/summary.md."""

from __future__ import annotations

import common


def f0(x) -> str:
    return "—" if x is None else f"{x:.0f}"


def main() -> None:
    cal = common.load_json("calibrate_hf.json", phase=2)
    win = common.load_json("hf_window.json", phase=2)
    ctl = common.load_json("validate_controller.json", phase=2)
    td = common.load_json("task_difficulty.json", phase=2)
    tdw = common.load_json("task_difficulty_wide.json", phase=2)
    tdp = common.load_json("task_difficulty_protocols.json", phase=2)
    L = ["# Fase 2: paciente, escenarios, entorno y líneas base", ""]

    # --- HF patient
    L += ["## Paciente con insuficiencia bajo LVAD (Fernández-Zapico 2024, Tabla 1)", ""]
    fits = cal["fits"]
    L += ["| Variable | Objetivo | Rs libre | Rs fijo 0.9 |", "| --- | --- | --- | --- |"]
    for k, (lo, hi) in cal["targets"].items():
        cells = [f"{fits[v]['metrics'][k]:.2f}{'' if fits[v]['in_range'][k] else ' ✗'}" for v in ("rs_free", "rs_fixed_0.9")]
        L.append(f"| {k} | {lo}–{hi} | {cells[0]} | {cells[1]} |")
    for v in ("rs_free", "rs_fixed_0.9"):
        p = fits[v]["params"]
        L.append(f"\n{v}: Emax {p['emax']:.3f}, Emin {p['emin']:.4f}, V0 {p['v0']:.1f} ml, Vtot {p['vtot']:.1f} ml, "
                 f"ω {p['omega']:.0f} rpm, Rs {p['rs']:.3f}; costo {fits[v]['cost']:.2e}; "
                 f"{sum(fits[v]['in_range'].values())}/8 en rango.")
    L += ["", f"Elegido: **{cal['chosen']}** (✗ = fuera de rango).", ""]

    # --- operating window
    L += ["## Ventana de operación del paciente (estado estacionario, velocidad constante)", "",
          "| Variación | ω reflujo > 1 ml | ω reflujo > 0 | ω cierre aórtico | ω succión | reflujo en ω cierre (ml/latido) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in win["rows"]:
        vb = r["v_back_at_avc_ml"]
        vb_txt = "—" if vb is None else f"{vb:.1f}"
        L.append(f"| {r['var']} = {r['value']} | {f0(r['omega_back_1.0ml'])} | {f0(r['omega_back_0.0ml'])} | {f0(r['omega_avc'])} | "
                 f"{f0(r['omega_suc'])} | {vb_txt} |")
    L += ["", "— en ω succión: no hay succión bajo 16 krpm. — en reflujo en ω cierre: no hay reflujo en esa velocidad.", ""]

    # --- controller validation
    L += ["## Controlador de Simaan (V6, V6b, V6c)", ""]
    L.append(f"- V6, publicado (c = 0.76, K = 152): diverge; velocidad máx. {ctl['v6_literal']['max_omega']:.0f} rpm, "
             f"{ctl['v6_literal']['beats_with_suction']} latidos con succión de 60.")
    v6 = ctl["v6"]
    L.append(f"- V6, c = 0.15 (K = 30): {v6['omega_at_10s']:.0f} rpm a los 10 s y {v6['omega_at_20s']:.0f} a los 20 s "
             f"(Fig. 16: ~15 000 y ~15 250); final {v6['omega_final']:.0f}; {v6['beats_with_suction']} latidos con succión.")
    L.append(f"- V6b, c = 0.15 sin límite: colapsa a {ctl['v6b']['omega_final']:.0f} rpm tras la rampa de Rs. "
             f"Con límite de 250 rpm/s: {ctl['v6b_slew']['omega_at_27s']:.0f} → {ctl['v6b_slew']['omega_final']:.0f} rpm "
             f"(Fig. 17: 15 250 → 14 250), {ctl['v6b_slew']['beats_with_suction']} latidos con succión.")
    L += ["", "| Rs | min-flow: ω, gasto, PAM | Tabla V | Δp: ω, gasto, PAM | Tabla V |", "| --- | --- | --- | --- | --- |"]
    for rs, row in ctl["v6c"].items():
        m, d = row["minflow"], row["deltap"]
        L.append(f"| {rs} | {m['omega']:.0f}, {m['co']:.2f}, {m['map']:.0f} | {m['table_v'][0]}, {m['table_v'][1]:.0f} | "
                 f"{d['omega']:.0f}, {d['co']:.2f}, {d['map']:.0f} | {d['table_v'][0]}, {d['table_v'][1]:.0f} |")
    L.append("")

    # --- task difficulty
    for title, d in (("rangos por defecto (relativos al paciente)", td), ("rangos amplios (Rs 0.5–2.0 absoluto)", tdw)):
        L += [f"## Dificultad: {d['n_episodes']} escenarios aleatorios de {d['episode_beats']} latidos, {title}", "",
              "| Política | Episodios con violación N=3 / 5 / 10 | Violaciones / 100 latidos (N=5) | AVO | SUC | BACK | LOW | Gasto medio | Retorno (N=5) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p, m in d["random_scenarios"].items():
            L.append(f"| {p} | {m['N3_episodes_with_violation']:.0%} / {m['N5_episodes_with_violation']:.0%} / {m['N10_episodes_with_violation']:.0%} | "
                     f"{m['N5_violations_per_100_beats']:.1f} | {m['avo_frac']:.0%} | {m['suc_frac']:.1%} | {m['back_frac']:.0%} | "
                     f"{m['low_frac']:.0%} | {m['co_mean']:.2f} | {m['return_mean_N5']:.0f} ± {m['return_std_N5']:.0f} |")
        L.append("")
    L += [f"## Protocolos de evaluación ({tdp['episode_beats']} latidos): violaciones / 100 latidos (N = 5)", "",
          "| Protocolo | " + " | ".join(next(iter(tdp["protocols"].values())).keys()) + " |",
          "| --- |" + " --- |" * len(next(iter(tdp["protocols"].values())))]
    for proto, row in tdp["protocols"].items():
        L.append(f"| {proto} | " + " | ".join(f"{m['N5_violations_per_100_beats']:.1f} (gasto {m['co_mean']:.1f})" for m in row.values()) + " |")
    L.append("")
    path = common.results_dir(2) / "summary.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
