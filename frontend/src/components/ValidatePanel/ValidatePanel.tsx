import { useCallback, useState } from "react";

type ValidationRow = {
  name: string;
  ok: boolean;
  status: number;
  ms: number;
  detail?: string;
};

function b64Src(b64: string | undefined): string | null {
  if (!b64) {
    return null;
  }
  return `data:image/png;base64,${b64}`;
}

export default function ValidatePanel() {
  const [busy, setBusy] = useState(false);
  const [rows, setRows] = useState<ValidationRow[]>([]);
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);
  const [maps, setMaps] = useState<Record<string, string> | null>(null);

  const runValidation = useCallback(async () => {
    setBusy(true);
    setError(null);
    setSummary(null);
    setMaps(null);
    setMetrics(null);
    try {
      const response = await fetch("/v1/validation/run", { method: "POST" });
      const data = await response.json();
      if (!response.ok && !data.stages) {
        throw new Error(data.message ?? "Validation request failed");
      }
      const nextRows: ValidationRow[] = (data.stages ?? []).map((stage: ValidationRow & { status?: number }) => ({
        name: stage.name,
        ok: Boolean(stage.ok),
        status: stage.status,
        ms: Number(stage.ms ?? 0),
        detail: stage.detail,
      }));
      setRows(nextRows);
      setSummary(data.summary ?? (data.ok ? "Validation passed" : "Validation failed"));
      setMetrics(data.metrics ?? null);
      setMaps(data.maps ?? null);
      if (!data.ok) {
        setError("One or more pipeline stages failed. See results.");
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Validation failed");
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <div className="grid h-full min-h-0 grid-cols-[380px_1fr] bg-slate-950 text-slate-100">
      <aside className="overflow-y-auto border-r border-slate-800 bg-slate-900 p-4 text-sm">
        <div className="space-y-4">
          <section>
            <div className="text-lg font-semibold text-white">Validation</div>
            <p className="mt-2 text-xs leading-5 text-slate-400">
              Live smoke tests for preview, tiled inference, and calibration, plus a synthetic DSM
              accuracy check (RMSE / MAE / R²).
            </p>
          </section>

          <section className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <button
              onClick={runValidation}
              disabled={busy}
              className="w-full rounded-md bg-cyan-500 px-3 py-2 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {busy ? "Running…" : "Run smoke validation"}
            </button>
          </section>

          {summary && (
            <section className={`rounded-lg border p-3 ${summary.includes("passed") ? "border-emerald-700/40 bg-emerald-950/30 text-emerald-200" : "border-rose-700/40 bg-rose-950/30 text-rose-200"}`}>
              {summary}
            </section>
          )}

          {error && (
            <section className="rounded-lg border border-rose-700/40 bg-rose-950/30 p-3 text-rose-200">
              {error}
            </section>
          )}

          {metrics && (
            <section className="rounded-lg border border-slate-800 bg-slate-950/70 p-3 text-xs text-slate-300">
              <div className="mb-2 font-semibold uppercase tracking-[0.2em] text-slate-500">Accuracy</div>
              <div>RMSE {metrics.rmse_m?.toFixed(2)} m</div>
              <div>MAE {metrics.mae_m?.toFixed(2)} m</div>
              <div>R² {metrics.r_squared?.toFixed(3)}</div>
              <div>r {metrics.correlation?.toFixed(3)}</div>
              <div>bias {metrics.bias_m?.toFixed(2)} m</div>
            </section>
          )}

          <section className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Results</div>
            {rows.length ? (
              <div className="space-y-2 text-xs">
                {rows.map((row) => (
                  <div key={row.name} className="flex items-center justify-between gap-3 rounded-md bg-slate-900 px-3 py-2">
                    <div>
                      <div className="font-medium text-slate-100">{row.name}</div>
                      <div className="text-slate-400">{row.detail ?? ""}</div>
                    </div>
                    <div className="text-right">
                      <div className={row.ok ? "text-emerald-300" : "text-rose-300"}>{row.ok ? "ok" : "fail"}</div>
                      <div className="text-slate-400">{row.status} · {row.ms.toFixed(0)} ms</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-slate-500">No validation run yet.</div>
            )}
          </section>
        </div>
      </aside>

      <main className="min-h-0 overflow-y-auto bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 p-8">
        <div className="text-sm uppercase tracking-[0.32em] text-cyan-300">Pipeline checks</div>
        <h2 className="mt-4 text-4xl font-semibold tracking-tight text-white">Validate the live backend</h2>
        <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
          Exercises preview, tiled prediction, and calibration, then recovers a known synthetic DSM
          so RMSE/MAE/correlation are reported against ground truth.
        </p>
        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          {(["reference_color_b64", "relative_color_b64", "predicted_color_b64"] as const).map((key) => (
            <div key={key} className="overflow-hidden rounded-2xl border border-slate-700 bg-slate-900/70">
              <div className="px-4 pt-4 text-xs uppercase tracking-[0.2em] text-slate-500">
                {key.replace("_color_b64", "").replace("_", " ")}
              </div>
              {b64Src(maps?.[key]) ? (
                <img src={b64Src(maps?.[key]) ?? ""} alt={key} className="mt-2 h-40 w-full object-cover" />
              ) : (
                <div className="mt-2 flex h-40 items-center justify-center text-xs text-slate-500">Run validation</div>
              )}
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
