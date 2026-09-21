import { create } from "zustand";

export type AppMode = "relative" | "absolute";

export interface AppState {
  mode: AppMode;
  dsmUrl: string | null;
  textureUrl: string | null;
  exaggeration: number;
  setMode: (mode: AppMode) => void;
  setDsmUrl: (dsmUrl: string | null) => void;
  setTextureUrl: (textureUrl: string | null) => void;
  setExaggeration: (exaggeration: number) => void;
}

export const useAppStore = create<AppState>((set) => ({
  mode: "relative",
  dsmUrl: null,
  textureUrl: null,
  exaggeration: 1,
  setMode: (mode) => set({ mode }),
  setDsmUrl: (dsmUrl) => set({ dsmUrl }),
  setTextureUrl: (textureUrl) => set({ textureUrl }),
  setExaggeration: (exaggeration) => set({ exaggeration }),
}));
