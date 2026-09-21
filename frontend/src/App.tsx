import Viewer3D from "./components/Viewer3D/Viewer3D";
import { useAppStore } from "./store/useAppStore";

export default function App() {
  const mode = useAppStore((s) => s.mode);
  const exaggeration = useAppStore((s) => s.exaggeration);

  return (
    <div className="flex h-full w-full flex-col bg-slate-950 text-slate-100">
      <header className="flex items-center justify-between border-b border-slate-800 px-4 py-2 text-sm">
        <span className="font-semibold tracking-wide">DepthWizard</span>
        <span className="text-slate-400">
          mode={mode} · exaggeration={exaggeration.toFixed(1)}×
        </span>
      </header>
      <div className="min-h-0 flex-1">
        <Viewer3D />
      </div>
    </div>
  );
}
