import { useEffect, useRef } from 'react';
import {
  Map as MapLibreMap,
  NavigationControl,
  setWorkerUrl,
  type GeoJSONSource,
  type StyleSpecification,
} from 'maplibre-gl';
import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import { bbox } from '@turf/turf';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useStore, type LayerKey } from '../store';
import type { FeatureCollection } from 'geojson';

setWorkerUrl(maplibreWorkerUrl);

const CLUSTER_COLORS = [
  '#4caf50',
  '#2196f3',
  '#ff9800',
  '#e91e63',
  '#9c27b0',
  '#00bcd4',
  '#8bc34a',
  '#ff5722',
  '#3f51b5',
  '#cddc39',
];

const EMPTY: FeatureCollection = { type: 'FeatureCollection', features: [] };

const STYLE: StyleSpecification = {
  version: 8,
  projection: { type: 'mercator' },
  sources: {
    esri: {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      attribution: 'Esri',
      maxzoom: 19,
    },
  },
  layers: [
    { id: 'background', type: 'background', paint: { 'background-color': '#0b1a12' } },
    { id: 'esri', type: 'raster', source: 'esri' },
  ],
};

const DAMAGE_COLOR: unknown = [
  'interpolate',
  ['linear'],
  ['coalesce', ['get', 'damage'], 0],
  0,
  '#228B22',
  25,
  '#9ACD32',
  50,
  '#FFD700',
  75,
  '#FF8C00',
  100,
  '#B22222',
];

function ensureSource(map: MapLibreMap, id: string, data: FeatureCollection) {
  const src = map.getSource(id) as GeoJSONSource | undefined;
  if (src) src.setData(data);
  else map.addSource(id, { type: 'geojson', data });
}

function whenReady(map: MapLibreMap, fn: () => void) {
  if (map.isStyleLoaded()) {
    fn();
    return;
  }
  map.once('load', fn);
}

function setVis(map: MapLibreMap, layerId: string, show: boolean) {
  if (!map.getLayer(layerId)) return;
  map.setLayoutProperty(layerId, 'visibility', show ? 'visible' : 'none');
}

function bringToFront(map: MapLibreMap, layerId: string) {
  if (map.getLayer(layerId)) map.moveLayer(layerId);
}

export function MapView() {
  const ref = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const readyRef = useRef(false);
  const lotFc = useStore((s) => s.lotFc);
  const samplesFc = useStore((s) => s.samplesFc);
  const centroidsFc = useStore((s) => s.centroidsFc);
  const clustersFc = useStore((s) => s.clustersFc);
  const idwOverlay = useStore((s) => s.idwOverlay);
  const layers = useStore((s) => s.layers);

  useEffect(() => {
    if (!ref.current || mapRef.current) return;
    const map = new MapLibreMap({
      container: ref.current,
      style: STYLE,
      center: [-62.1, -32.7],
      zoom: 7,
      attributionControl: false,
    });
    map.addControl(new NavigationControl({ visualizePitch: false }), 'top-right');
    mapRef.current = map;
    (window as unknown as { __portalMap?: MapLibreMap }).__portalMap = map;

    const onResize = () => map.resize();
    window.addEventListener('resize', onResize);

    map.on('load', () => {
      readyRef.current = true;
      map.resize();
      // Base vector sources so later setData always works.
      ensureSource(map, 'lot', EMPTY);
      ensureSource(map, 'samples', EMPTY);
      ensureSource(map, 'clusters', EMPTY);
      ensureSource(map, 'centroids', EMPTY);

      map.addLayer({
        id: 'lot-fill',
        type: 'fill',
        source: 'lot',
        paint: { 'fill-color': '#fbbf24', 'fill-opacity': 0.28 },
      });
      map.addLayer({
        id: 'lot-line',
        type: 'line',
        source: 'lot',
        paint: { 'line-color': '#ef4444', 'line-width': 2.5 },
      });

      const matchExpr: unknown[] = ['match', ['%', ['coalesce', ['get', 'cluster'], ['get', 'ambiente'], 0], 10]];
      CLUSTER_COLORS.forEach((c, i) => {
        matchExpr.push(i, c);
      });
      matchExpr.push('#888888');
      map.addLayer({
        id: 'clusters-fill',
        type: 'fill',
        source: 'clusters',
        paint: { 'fill-color': matchExpr as never, 'fill-opacity': 0.32 },
      });
      map.addLayer({
        id: 'clusters-line',
        type: 'line',
        source: 'clusters',
        paint: { 'line-color': '#ffffff', 'line-width': 1.2 },
      });

      map.addLayer({
        id: 'samples-circle',
        type: 'circle',
        source: 'samples',
        paint: {
          'circle-radius': 7,
          'circle-color': DAMAGE_COLOR as never,
          'circle-stroke-width': 1.5,
          'circle-stroke-color': '#ffffff',
        },
      });

      map.addLayer({
        id: 'centroids-circle',
        type: 'circle',
        source: 'centroids',
        paint: {
          'circle-radius': 10,
          'circle-color': [
            'case',
            ['has', 'damage'],
            DAMAGE_COLOR,
            '#00e5ff',
          ] as never,
          'circle-stroke-width': 2,
          'circle-stroke-color': '#0c1410',
        },
      });
      map.addLayer({
        id: 'centroids-label',
        type: 'symbol',
        source: 'centroids',
        layout: {
          'text-field': ['coalesce', ['get', 'label'], ['concat', 'C', ['+', ['get', 'id'], 1]]],
          'text-size': 11,
          'text-offset': [0, 1.35],
        },
        paint: { 'text-color': '#ffffff', 'text-halo-color': '#000', 'text-halo-width': 1.2 },
      });
    });

    return () => {
      window.removeEventListener('resize', onResize);
      readyRef.current = false;
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Lot polygon + camera
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      ensureSource(map, 'lot', lotFc ?? EMPTY);
      if (lotFc?.features?.length) {
        try {
          const b = bbox(lotFc);
          map.fitBounds(
            [
              [b[0], b[1]],
              [b[2], b[3]],
            ],
            { padding: 56, duration: 700, maxZoom: 16 },
          );
        } catch {
          /* ignore */
        }
      }
      map.resize();
      map.triggerRepaint();
    };
    whenReady(map, apply);
  }, [lotFc]);

  // Samples / centroids / clusters data
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      ensureSource(map, 'samples', samplesFc ?? EMPTY);
      ensureSource(map, 'centroids', centroidsFc ?? EMPTY);
      ensureSource(map, 'clusters', clustersFc ?? EMPTY);
      bringToFront(map, 'samples-circle');
      bringToFront(map, 'centroids-circle');
      bringToFront(map, 'centroids-label');
      map.triggerRepaint();
    };
    whenReady(map, apply);
  }, [samplesFc, centroidsFc, clustersFc]);

  // IDW raster overlay
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      if (!idwOverlay) {
        if (map.getLayer('idw-layer')) map.removeLayer('idw-layer');
        if (map.getSource('idw')) map.removeSource('idw');
        return;
      }
      const coords = [
        [idwOverlay.bounds[0], idwOverlay.bounds[3]],
        [idwOverlay.bounds[2], idwOverlay.bounds[3]],
        [idwOverlay.bounds[2], idwOverlay.bounds[1]],
        [idwOverlay.bounds[0], idwOverlay.bounds[1]],
      ];
      if (map.getLayer('idw-layer')) map.removeLayer('idw-layer');
      if (map.getSource('idw')) map.removeSource('idw');
      map.addSource('idw', { type: 'image', url: idwOverlay.url, coordinates: coords });
      map.addLayer(
        {
          id: 'idw-layer',
          type: 'raster',
          source: 'idw',
          paint: { 'raster-opacity': 0.55 },
        },
        'samples-circle',
      );
      map.triggerRepaint();
    };
    whenReady(map, apply);
  }, [idwOverlay]);

  // Layer visibility
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => {
      const vis: Record<string, LayerKey> = {
        'lot-fill': 'lot',
        'lot-line': 'lot',
        'samples-circle': 'samples',
        'centroids-circle': 'centroids',
        'centroids-label': 'centroids',
        'clusters-fill': 'clusters',
        'clusters-line': 'clusters',
        'idw-layer': 'idw',
      };
      for (const [layerId, key] of Object.entries(vis)) {
        setVis(map, layerId, !!layers[key]);
      }
    };
    whenReady(map, apply);
  }, [layers, samplesFc, centroidsFc, clustersFc, idwOverlay, lotFc]);

  return <div ref={ref} className="w-full h-full min-h-[420px]" />;
}
