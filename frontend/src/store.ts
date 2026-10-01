import { create } from 'zustand';
import type { CaseItem, InspectResult } from './lib/api';
import type { FeatureCollection } from 'geojson';

export type LayerKey = 'lot' | 'samples' | 'centroids' | 'clusters' | 'idw';

type Store = {
  cases: CaseItem[];
  selected: CaseItem | null;
  detail: Record<string, unknown> | null;
  result: InspectResult | null;
  lotFc: FeatureCollection | null;
  samplesFc: FeatureCollection | null;
  centroidsFc: FeatureCollection | null;
  clustersFc: FeatureCollection | null;
  idwOverlay: { url: string; bounds: [number, number, number, number] } | null;
  layers: Record<LayerKey, boolean>;
  loading: boolean;
  error: string | null;
  actaClosed: boolean;
  setCases: (c: CaseItem[]) => void;
  setSelected: (c: CaseItem | null) => void;
  setDetail: (d: Record<string, unknown> | null) => void;
  setResult: (r: InspectResult | null) => void;
  setLotFc: (fc: FeatureCollection | null) => void;
  setSamplesFc: (fc: FeatureCollection | null) => void;
  setCentroidsFc: (fc: FeatureCollection | null) => void;
  setClustersFc: (fc: FeatureCollection | null) => void;
  setIdwOverlay: (o: { url: string; bounds: [number, number, number, number] } | null) => void;
  setLayers: (partial: Partial<Record<LayerKey, boolean>>) => void;
  toggleLayer: (k: LayerKey) => void;
  setLoading: (v: boolean) => void;
  setError: (e: string | null) => void;
  setActaClosed: (v: boolean) => void;
  resetMapLayers: () => void;
};

const defaultLayers: Record<LayerKey, boolean> = {
  lot: true,
  samples: true,
  centroids: true,
  clusters: true,
  idw: true,
};

export const useStore = create<Store>((set) => ({
  cases: [],
  selected: null,
  detail: null,
  result: null,
  lotFc: null,
  samplesFc: null,
  centroidsFc: null,
  clustersFc: null,
  idwOverlay: null,
  layers: { ...defaultLayers },
  loading: false,
  error: null,
  actaClosed: false,
  setCases: (cases) => set({ cases }),
  setSelected: (selected) =>
    set({
      selected,
      actaClosed: false,
      result: null,
      centroidsFc: null,
      clustersFc: null,
      idwOverlay: null,
    }),
  setDetail: (detail) => set({ detail }),
  setResult: (result) => set({ result }),
  setLotFc: (lotFc) => set({ lotFc }),
  setSamplesFc: (samplesFc) => set({ samplesFc }),
  setCentroidsFc: (centroidsFc) => set({ centroidsFc }),
  setClustersFc: (clustersFc) => set({ clustersFc }),
  setIdwOverlay: (idwOverlay) => set({ idwOverlay }),
  setLayers: (partial) => set((s) => ({ layers: { ...s.layers, ...partial } })),
  toggleLayer: (k) => set((s) => ({ layers: { ...s.layers, [k]: !s.layers[k] } })),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setActaClosed: (actaClosed) => set({ actaClosed }),
  resetMapLayers: () =>
    set({
      samplesFc: null,
      centroidsFc: null,
      clustersFc: null,
      idwOverlay: null,
      result: null,
    }),
}));
