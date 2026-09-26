import { create } from "zustand";

export type AppMode = "relative" | "absolute";
export type WorkspaceTab = "validate" | "sculpt";

export interface AppState {
  mode: AppMode;
  workspaceTab: WorkspaceTab;
  activeLayer: "image" | "relative" | "calibrated";
  dsmUrl: string | null;
  textureUrl: string | null;
  exaggeration: number;
  setMode: (mode: AppMode) => void;
  setWorkspaceTab: (workspaceTab: WorkspaceTab) => void;
  setActiveLayer: (activeLayer: "image" | "relative" | "calibrated") => void;
  setDsmUrl: (dsmUrl: string | null) => void;
  setTextureUrl: (textureUrl: string | null) => void;
  setExaggeration: (exaggeration: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  mode: "relative",
  workspaceTab: "sculpt",
  activeLayer: "relative",
  dsmUrl: null,
  textureUrl: null,
  exaggeration: 1,
  setMode: (mode) => set({ mode }),
  setWorkspaceTab: (workspaceTab) => set({ workspaceTab }),
  setActiveLayer: (activeLayer) => set({ activeLayer }),
  setDsmUrl: (dsmUrl) => set({ dsmUrl }),
  setTextureUrl: (textureUrl) => set({ textureUrl }),
  setExaggeration: (exaggeration) => set({ exaggeration }),
}));
