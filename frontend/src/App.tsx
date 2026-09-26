import Viewer3D from "./components/Viewer3D/Viewer3D";
import ValidatePanel from "./components/ValidatePanel/ValidatePanel";
import { useAppStore } from "./store/useAppStore";

export default function App() {
  const mode = useAppStore((s) => s.mode);
  const exaggeration = useAppStore((s) => s.exaggeration);
  const activeLayer = useAppStore((s) => s.activeLayer);
  const workspaceTab = useAppStore((s) => s.workspaceTab);
  const setWorkspaceTab = useAppStore((s) => s.setWorkspaceTab);

  return (
    <div className="flex h-full w-full flex-col bg-slate-950 text-slate-100">
      <header className="flex items-center justify-between border-b border-slate-800 px-4 py-2 text-sm">
        <div className="flex items-center gap-3">
          <span className="font-semibold tracking-wide">DepthWizard</span>
          <div className="flex rounded-full border border-slate-700 bg-slate-900 p-1 text-xs">
            <button
              onClick={() => setWorkspaceTab("validate")}
              className={`rounded-full px-3 py-1 transition ${workspaceTab === "validate" ? "bg-cyan-500 text-slate-950" : "text-slate-300 hover:bg-slate-800"}`}
            >
              Validate
            </button>
            <button
              onClick={() => setWorkspaceTab("sculpt")}
              className={`rounded-full px-3 py-1 transition ${workspaceTab === "sculpt" ? "bg-cyan-500 text-slate-950" : "text-slate-300 hover:bg-slate-800"}`}
            >
              Sculpt
            </button>
          </div>
        </div>
        <span className="text-slate-400">
          layer={activeLayer} · mode={mode} · exaggeration={exaggeration.toFixed(1)}×
        </span>
      </header>
      <div className="min-h-0 flex-1">
        {workspaceTab === "validate" ? <ValidatePanel /> : <Viewer3D />}
      </div>
    </div>
  );
}
