"""
summarize_epsilon_sensitivity_v3c.py

Summarize epsilon sensitivity output.

Input:
    outputs/epsilon_sensitivity_v3c_100k_5seeds.xlsx

Output:
    outputs/epsilon_sensitivity_v3c_summary.xlsx
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def find_file(filename: str) -> Path:
    candidates = [
        Path.cwd() / "outputs" / filename,
        Path.cwd() / filename,
        Path(__file__).resolve().parent / "outputs" / filename,
        Path(__file__).resolve().parent / filename,
        Path("/mnt/data") / "outputs" / filename,
        Path("/mnt/data") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked:\n{checked}")


INPUT_FILE = find_file("epsilon_sensitivity_v3c_100k_5seeds.xlsx")
OUTPUT_FILE = INPUT_FILE.with_name("epsilon_sensitivity_v3c_summary.xlsx")


def main() -> None:
    summary = pd.read_excel(INPUT_FILE, sheet_name="Scenario_Summary")
    config = pd.read_excel(INPUT_FILE, sheet_name="Scenario_Config")
    metrics_seed = pd.read_excel(INPUT_FILE, sheet_name="Metrics_By_Seed")

    test = summary[summary["SPLIT"].astype(str).str.lower().eq("test")].copy()

    # Sort by algorithm then epsilon.
    if "EPSILON" in test.columns:
        test = test.sort_values(["ALGORITHM", "EPSILON"])
    else:
        test = test.sort_values(["ALGORITHM", "VARIANT"])

    # Best epsilon by algorithm for different criteria.
    best_rows = []
    for algo, g in test.groupby("ALGORITHM"):
        for criterion, ascending in [
            ("MEAN_OF_MEAN_PNL", False),
            ("MEAN_OF_CVAR_95", False),
            ("MEAN_OF_SHARPE_LIKE", False),
            ("MEAN_TC", True),
            ("MEAN_TURNOVER", True),
        ]:
            if criterion in g.columns and len(g) > 0:
                row = g.sort_values(criterion, ascending=ascending).iloc[0].copy()
                row["BEST_BY"] = criterion
                best_rows.append(row)
    best_by_criterion = pd.DataFrame(best_rows)

    # Add adjacent deltas by algorithm, useful for choosing epsilon without over-reading.
    delta_frames = []
    numeric_cols = [
        "MEAN_OF_MEAN_PNL",
        "MEAN_OF_CVAR_95",
        "MEAN_OF_SHARPE_LIKE",
        "MEAN_TC",
        "MEAN_TURNOVER",
        "AVG_ADJUSTMENT_FROM_DELTA",
        "NO_TRADE_RATE",
    ]
    numeric_cols = [c for c in numeric_cols if c in test.columns]
    if "EPSILON" in test.columns:
        for algo, g in test.groupby("ALGORITHM"):
            g = g.sort_values("EPSILON").copy()
            for c in numeric_cols:
                g[f"DELTA_{c}_vs_prev_eps"] = g[c].diff()
            delta_frames.append(g)
        test_with_deltas = pd.concat(delta_frames, ignore_index=True)
    else:
        test_with_deltas = test.copy()

    # Interpretation flags
    flags = []
    for _, row in test.iterrows():
        notes = []
        eps = row.get("EPSILON", None)
        if pd.notna(eps):
            if eps <= 0.05:
                notes.append("tight residual bound")
            elif eps >= 0.20:
                notes.append("wide residual bound")
        if row.get("AVG_ADJUSTMENT_FROM_DELTA", 0) > 0.12:
            notes.append("larger residual adjustment")
        if row.get("MEAN_OF_CVAR_95", -1e9) < -500:
            notes.append("weak tail-risk region")
        if row.get("MEAN_TC", 1e9) > 70:
            notes.append("high transaction cost")
        flags.append("; ".join(notes) if notes else "balanced / no major flag")
    test["INTERPRETATION_FLAGS"] = flags

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        test.to_excel(writer, sheet_name="Test_Epsilon_Summary", index=False)
        test_with_deltas.to_excel(writer, sheet_name="Adjacent_Epsilon_Deltas", index=False)
        best_by_criterion.to_excel(writer, sheet_name="Best_By_Criterion", index=False)
        summary.to_excel(writer, sheet_name="Full_Summary", index=False)
        config.to_excel(writer, sheet_name="Scenario_Config", index=False)
        metrics_seed.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)

    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    cols = [
        "ALGORITHM", "VARIANT", "EPSILON",
        "MEAN_OF_MEAN_PNL", "MEAN_OF_CVAR_95", "MEAN_OF_SHARPE_LIKE",
        "MEAN_TC", "MEAN_TURNOVER", "AVG_ADJUSTMENT_FROM_DELTA", "NO_TRADE_RATE"
    ]
    cols = [c for c in cols if c in test.columns]
    print("\nTest epsilon summary:")
    print(test[cols])


if __name__ == "__main__":
    main()
