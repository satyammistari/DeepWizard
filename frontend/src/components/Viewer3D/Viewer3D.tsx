import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { useAppStore } from "../../store/useAppStore";

const MAX_MESH = 192;

type HeightField = { width: number; height: number; pixels: Float32Array };

type PreviewMaps = {
  relativeGray: string;
  relativeColor: string;
  overlay: string;
};

type CalibratedMaps = {
  relativeColor: string;
  calibratedColor: string;
  hillshade: string;
  uncertainty: string;
  overlay: string;
  calibratedGray: string;
};

function b64ToObjectUrl(b64: string): string {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
}

function downsample(field: HeightField, maxDim: number): HeightField {
  const { width, height, pixels } = field;
  const scale = Math.max(width, height) / maxDim;
  if (scale <= 1) {
    return field;
  }
  const w = Math.max(2, Math.round(width / scale));
  const h = Math.max(2, Math.round(height / scale));
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y += 1) {
    const srcY = Math.min(height - 1, Math.floor((y / (h - 1)) * (height - 1)));
    for (let x = 0; x < w; x += 1) {
      const srcX = Math.min(width - 1, Math.floor((x / (w - 1)) * (width - 1)));
      out[y * w + x] = pixels[srcY * width + srcX];
    }
  }
  return { width: w, height: h, pixels: out };
}

function TerrainMesh({
  heightData,
  textureUrl,
  width = 100,
  height = 100,
  exaggeration = 1,
}: {
  heightData: HeightField | null;
  textureUrl: string | null;
  width?: number;
  height?: number;
  exaggeration?: number;
}) {
  const ref = useRef<THREE.Mesh>(null);
  const meshData = useMemo(
    () => (heightData ? downsample(heightData, MAX_MESH) : null),
    [heightData],
  );

  const geom = useMemo(() => {
    const cols = meshData?.width ?? 64;
    const rows = meshData?.height ?? 64;
    const geometry = new THREE.PlaneGeometry(width, height, cols - 1, rows - 1);
    if (meshData?.pixels) {
      const pixels = meshData.pixels;
      const count = geometry.attributes.position.count;
      for (let i = 0; i < count; i += 1) {
        const z = pixels[Math.min(i, pixels.length - 1)] * 10.0 * exaggeration;
        geometry.attributes.position.setZ(i, z - 5.0);
      }
      geometry.computeVertexNormals();
      geometry.attributes.position.needsUpdate = true;
    }
    return geometry;
  }, [meshData, width, height, exaggeration]);

  const texture = useMemo(() => {
    if (!textureUrl) {
      return null;
    }
    const loaded = new THREE.TextureLoader().load(textureUrl);
    loaded.colorSpace = THREE.SRGBColorSpace;
    return loaded;
  }, [textureUrl]);

  useEffect(() => {
    return () => {
      geom.dispose();
      texture?.dispose();
    };
  }, [geom, texture]);

  return (
    <mesh ref={ref} rotation={[-Math.PI / 2, 0, 0]} geometry={geom}>
      <meshStandardMaterial map={texture} color={texture ? "#ffffff" : "#3f7f5f"} metalness={0} roughness={1} />
    </mesh>
  );
}

async function decodeHeightImage(url: string): Promise<HeightField | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const w = img.width;
      const h = img.height;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve(null);
        return;
      }
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, w, h).data;
      const pixels = new Float32Array(w * h);
      for (let i = 0; i < w * h; i += 1) {
        pixels[i] = data[i * 4] / 255.0;
      }
      resolve({ width: w, height: h, pixels });
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function MapThumb({ label, src }: { label: string; src: string | null }) {
  return (
    <div className="overflow-hidden rounded-md border border-slate-800 bg-slate-950">
      <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-slate-400">{label}</div>
      {src ? (
        <img src={src} alt={label} className="h-24 w-full object-cover" />
      ) : (
        <div className="flex h-24 items-center justify-center text-[10px] text-slate-600">No map</div>
      )}
    </div>
  );
}

export default function Viewer3D() {
  const [file, setFile] = useState<File | null>(null);
  const [heightData, setHeightData] = useState<HeightField | null>(null);
  const [calibratedHeight, setCalibratedHeight] = useState<HeightField | null>(null);
  const [textureUrl, setTextureUrl] = useState<string | null>(null);
  const [previewMaps, setPreviewMaps] = useState<PreviewMaps | null>(null);
  const [calibratedMaps, setCalibratedMaps] = useState<CalibratedMaps | null>(null);
  const [calibrationBusy, setCalibrationBusy] = useState(false);
  const [calibrationError, setCalibrationError] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);
  const [calibrationInfo, setCalibrationInfo] = useState<string | null>(null);
  const [sunAzimuthDeg, setSunAzimuthDeg] = useState("90");
  const [sunElevationDeg, setSunElevationDeg] = useState("45");
  const [gsdMeters, setGsdMeters] = useState("1");
  const [bboxMinX, setBboxMinX] = useState("77.55");
  const [bboxMinY, setBboxMinY] = useState("12.90");
  const [bboxMaxX, setBboxMaxX] = useState("77.70");
  const [bboxMaxY, setBboxMaxY] = useState("13.02");
  const [demSource, setDemSource] = useState("SRTMGL1");
  const [busy, setBusy] = useState(false);
  const mode = useAppStore((state) => state.mode);
  const activeLayer = useAppStore((state) => state.activeLayer);
  const exaggeration = useAppStore((state) => state.exaggeration);
  const setMode = useAppStore((state) => state.setMode);
  const setActiveLayer = useAppStore((state) => state.setActiveLayer);
  const setExaggeration = useAppStore((state) => state.setExaggeration);

  const revoke = useCallback((url: string | null) => {
    if (url) {
      URL.revokeObjectURL(url);
    }
  }, []);

  const onFile = useCallback(
    (next: File | null) => {
      setFile(next);
      setTextureUrl((current) => {
        revoke(current);
        return next ? URL.createObjectURL(next) : null;
      });
      setPreviewMaps((current) => {
        if (current) {
          revoke(current.relativeGray);
          revoke(current.relativeColor);
          revoke(current.overlay);
        }
        return null;
      });
      setCalibratedMaps((current) => {
        if (current) {
          Object.values(current).forEach(revoke);
        }
        return null;
      });
      setCalibratedHeight(null);
      setHeightData(null);
      setCalibrationError(null);
      setPreviewError(null);
      setMetrics(null);
      setCalibrationInfo(null);
    },
    [revoke],
  );

  useEffect(() => {
    if (!file) {
      return;
    }
    let mounted = true;
    setBusy(true);
    setPreviewError(null);
    const fd = new FormData();
    fd.append("file", file, file.name);
    fetch("/v1/depth/maps", { method: "POST", body: fd })
      .then(async (resp) => {
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ message: `Preview failed (${resp.status})` }));
          throw new Error(err.message ?? `Preview failed (${resp.status})`);
        }
        return resp.json();
      })
      .then(async (data) => {
        if (!mounted) {
          return;
        }
        const maps: PreviewMaps = {
          relativeGray: b64ToObjectUrl(data.relative_gray_b64),
          relativeColor: b64ToObjectUrl(data.relative_color_b64),
          overlay: b64ToObjectUrl(data.overlay_b64),
        };
        setPreviewMaps(maps);
        const heights = await decodeHeightImage(maps.relativeGray);
        if (mounted) {
          setHeightData(heights);
        }
      })
      .catch((err: Error) => {
        if (mounted) {
          setPreviewError(err.message);
        }
      })
      .finally(() => {
        if (mounted) {
          setBusy(false);
        }
      });
    return () => {
      mounted = false;
    };
  }, [file]);

  const runCalibration = useCallback(async () => {
    if (!file) {
      setCalibrationError("Upload an image first.");
      return;
    }
    setCalibrationBusy(true);
    setCalibrationError(null);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      const params = new URLSearchParams({
        sun_azimuth_deg: sunAzimuthDeg,
        sun_elevation_deg: sunElevationDeg,
        gsd_m: gsdMeters,
        minx: bboxMinX,
        miny: bboxMinY,
        maxx: bboxMaxX,
        maxy: bboxMaxY,
        dem_source: demSource,
        cache_dir: ".dem_cache",
      });
      const response = await fetch(`/v1/calibration/calibrate_depth?${params.toString()}`, {
        method: "POST",
        body: fd,
      });
      const data = await response.json();
      if (!response.ok || data.status !== "success") {
        throw new Error(data.message ?? "Calibration failed");
      }
      const nextMaps: CalibratedMaps = {
        relativeColor: b64ToObjectUrl(data.maps.relative_color_b64),
        calibratedColor: b64ToObjectUrl(data.maps.calibrated_color_b64),
        hillshade: b64ToObjectUrl(data.maps.hillshade_b64),
        uncertainty: b64ToObjectUrl(data.maps.uncertainty_b64),
        overlay: b64ToObjectUrl(data.maps.overlay_b64),
        calibratedGray: b64ToObjectUrl(data.maps.calibrated_gray_b64),
      };
      setCalibratedMaps((current) => {
        if (current) {
          Object.values(current).forEach(revoke);
        }
        return nextMaps;
      });
      const heights = await decodeHeightImage(nextMaps.calibratedGray);
      setCalibratedHeight(heights);
      setActiveLayer("calibrated");
      setMetrics(data.metrics ?? null);
      const cal = data.calibration;
      const fallback = cal?.used_dem_fallback ? " · offline DEM prior" : "";
      setCalibrationInfo(
        `${cal?.dem_source ?? demSource} · ${cal?.method ?? "fit"} · scale=${Number(cal?.scale ?? 0).toFixed(2)} · ${cal?.vertical_datum ?? "EGM96"}${fallback}`,
      );
    } catch (error) {
      setCalibrationError(error instanceof Error ? error.message : "Calibration failed");
    } finally {
      setCalibrationBusy(false);
    }
  }, [
    bboxMaxX,
    bboxMaxY,
    bboxMinX,
    bboxMinY,
    demSource,
    file,
    gsdMeters,
    revoke,
    setActiveLayer,
    sunAzimuthDeg,
    sunElevationDeg,
  ]);

  const terrainTexture =
    activeLayer === "calibrated"
      ? calibratedMaps?.overlay ?? textureUrl
      : activeLayer === "relative"
        ? previewMaps?.overlay ?? textureUrl
        : textureUrl;

  return (
    <div className="flex h-full w-full">
      <div className="w-[26rem] overflow-y-auto border-r border-slate-800 bg-slate-900 p-3 text-sm text-slate-100">
        <div className="space-y-4">
          <section>
            <label className="mb-2 block font-medium">Upload image</label>
            <input
              aria-label="Upload image"
              onChange={(e) => onFile(e.target.files ? e.target.files[0] : null)}
              type="file"
              accept="image/*,.tif,.tiff"
              className="block w-full text-sm text-slate-300 file:mr-3 file:rounded-md file:border-0 file:bg-cyan-500 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-slate-950 hover:file:bg-cyan-400"
            />
            <div className="mt-2 text-xs text-slate-400">
              Generates relative depth, color maps, then a metric DSM with uncertainty.
            </div>
          </section>

          <section className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
            <div className="mb-2 font-medium">Viewer tools</div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <button className={`rounded-md px-2 py-2 ${activeLayer === "image" ? "bg-cyan-500 text-slate-950" : "bg-slate-800 text-slate-200"}`} onClick={() => setActiveLayer("image")}>Image</button>
              <button className={`rounded-md px-2 py-2 ${activeLayer === "relative" ? "bg-cyan-500 text-slate-950" : "bg-slate-800 text-slate-200"}`} onClick={() => setActiveLayer("relative")}>Relative</button>
              <button className={`rounded-md px-2 py-2 ${activeLayer === "calibrated" ? "bg-cyan-500 text-slate-950" : "bg-slate-800 text-slate-200"}`} onClick={() => setActiveLayer("calibrated")}>Calibrated</button>
            </div>
            <div className="mt-3 flex items-center justify-between gap-3">
              <span className="text-xs text-slate-400">Vertical exaggeration</span>
              <input
                type="range"
                min="0.5"
                max="8"
                step="0.1"
                value={exaggeration}
                onChange={(e) => setExaggeration(Number(e.target.value))}
                className="w-40"
              />
            </div>
            <div className="mt-3 flex items-center justify-between gap-3 text-xs">
              <span className="text-slate-400">Mode</span>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as "relative" | "absolute")}
                className="rounded bg-slate-800 px-2 py-1 text-slate-100"
              >
                <option value="relative">Relative</option>
                <option value="absolute">Absolute</option>
              </select>
            </div>
          </section>

          <section className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
            <div className="mb-2 font-medium">Preview maps</div>
            <div className="grid grid-cols-2 gap-2">
              <MapThumb label="Relative" src={previewMaps?.relativeColor ?? null} />
              <MapThumb label="Overlay" src={previewMaps?.overlay ?? null} />
              <MapThumb label="Calibrated" src={calibratedMaps?.calibratedColor ?? null} />
              <MapThumb label="Hillshade" src={calibratedMaps?.hillshade ?? null} />
              <MapThumb label="Uncertainty" src={calibratedMaps?.uncertainty ?? null} />
              <MapThumb label="RGB" src={textureUrl} />
            </div>
          </section>

          <section className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
            <div className="mb-2 font-medium">Calibration</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <input className="rounded bg-slate-800 px-2 py-1" value={sunAzimuthDeg} onChange={(e) => setSunAzimuthDeg(e.target.value)} placeholder="Sun azimuth" />
              <input className="rounded bg-slate-800 px-2 py-1" value={sunElevationDeg} onChange={(e) => setSunElevationDeg(e.target.value)} placeholder="Sun elevation" />
              <input className="rounded bg-slate-800 px-2 py-1" value={gsdMeters} onChange={(e) => setGsdMeters(e.target.value)} placeholder="GSD m" />
              <select className="rounded bg-slate-800 px-2 py-1" value={demSource} onChange={(e) => setDemSource(e.target.value)}>
                <option value="SRTMGL1">SRTMGL1</option>
                <option value="COP30">COP30</option>
                <option value="CARTODEM">CARTODEM</option>
              </select>
              <input className="rounded bg-slate-800 px-2 py-1" value={bboxMinX} onChange={(e) => setBboxMinX(e.target.value)} placeholder="min lon" />
              <input className="rounded bg-slate-800 px-2 py-1" value={bboxMinY} onChange={(e) => setBboxMinY(e.target.value)} placeholder="min lat" />
              <input className="rounded bg-slate-800 px-2 py-1" value={bboxMaxX} onChange={(e) => setBboxMaxX(e.target.value)} placeholder="max lon" />
              <input className="rounded bg-slate-800 px-2 py-1" value={bboxMaxY} onChange={(e) => setBboxMaxY(e.target.value)} placeholder="max lat" />
            </div>
            <button
              onClick={runCalibration}
              disabled={calibrationBusy || !file}
              className="mt-3 w-full rounded-md bg-emerald-500 px-3 py-2 font-semibold text-slate-950 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {calibrationBusy ? "Calibrating…" : "Run calibration"}
            </button>
            {calibrationInfo && <div className="mt-2 text-xs text-emerald-300">{calibrationInfo}</div>}
            {calibrationError && <div className="mt-2 text-xs text-rose-300">{calibrationError}</div>}
            {metrics && (
              <div className="mt-2 grid grid-cols-2 gap-1 text-[11px] text-slate-300">
                <div>RMSE {metrics.rmse_m?.toFixed(2)} m</div>
                <div>MAE {metrics.mae_m?.toFixed(2)} m</div>
                <div>R² {metrics.r_squared?.toFixed(3)}</div>
                <div>r {metrics.correlation?.toFixed(3)}</div>
              </div>
            )}
          </section>

          <section className="text-xs text-slate-400">
            {busy && <div className="text-yellow-300">Generating relative preview…</div>}
            {previewError && <div className="text-rose-300">{previewError}</div>}
            {calibratedMaps && <div className="text-emerald-300">Calibrated DSM ready.</div>}
          </section>
        </div>
      </div>
      <div className="relative flex-1">
        <Canvas camera={{ position: [80, 80, 80], fov: 50 }} className="h-full w-full">
          <color attach="background" args={["#0b1220"]} />
          <ambientLight intensity={0.8} />
          <directionalLight position={[60, 80, 40]} intensity={1.1} />
          <gridHelper args={[100, 20, "#64748b", "#1e293b"]} />
          <OrbitControls makeDefault />
          {activeLayer === "relative" && heightData && (
            <TerrainMesh heightData={heightData} textureUrl={terrainTexture} width={100} height={100} exaggeration={exaggeration} />
          )}
          {activeLayer === "image" && textureUrl && (
            <TerrainMesh heightData={heightData ?? { width: 2, height: 2, pixels: new Float32Array([0, 0, 0, 0]) }} textureUrl={textureUrl} width={100} height={100} exaggeration={0.01} />
          )}
          {activeLayer === "calibrated" && (calibratedHeight || heightData) && (
            <TerrainMesh
              heightData={calibratedHeight ?? heightData}
              textureUrl={terrainTexture}
              width={100}
              height={100}
              exaggeration={exaggeration}
            />
          )}
        </Canvas>
      </div>
    </div>
  );
}
