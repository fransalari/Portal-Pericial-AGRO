import { useEffect, useState } from 'react';
import clsx from 'clsx';
import { api, type CaseItem, type InspectResult } from './lib/api';
import { useStore, type LayerKey } from './store';
import { MapView } from './components/MapView';
import type { FeatureCollection } from 'geojson';

async function fetchFc(url: string): Promise<FeatureCollection | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as FeatureCollection;
  } catch {
    return null;
  }
}

export default function App() {
  const {
    cases,
    selected,
    detail,
    result,
    loading,
    error,
    actaClosed,
    layers,
    setCases,
    setSelected,
    setDetail,
    setResult,
    setLotFc,
    setSamplesFc,
    setCentroidsFc,
    setClustersFc,
    setIdwOverlay,
    setLayers,
    toggleLayer,
    setLoading,
    setError,
    setActaClosed,
  } = useStore();
  const [q, setQ] = useState('');
  const [method, setMethod] = useState<'traditional' | 'directed' | 'both'>('both');

  useEffect(() => {
    void loadCases();
  }, []);

  async function loadCases(query?: string) {
    setLoading(true);
    setError(null);
    try {
      const data = await api.cases(query);
      setCases(data.cases);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function openCase(c: CaseItem) {
    setSelected(c);
    setActaClosed(false);
    setResult(null);
    setCentroidsFc(null);
    setClustersFc(null);
    setIdwOverlay(null);
    setLoading(true);
    setError(null);
    try {
      const d = await api.caseDetail(c.lot_id);
      setDetail(d);
      const feature = d.feature as FeatureCollection['features'][0];
      setLotFc({ type: 'FeatureCollection', features: [feature] });
      const samples = d.samples_geojson as FeatureCollection | undefined;
      if (samples?.features?.length) {
        // Ensure numeric damage for MapLibre interpolate
        const normalized: FeatureCollection = {
          type: 'FeatureCollection',
          features: samples.features.map((f, i) => ({
            ...f,
            properties: {
              ...(f.properties || {}),
              id: i,
              damage: Number((f.properties as { damage?: number } | null)?.damage ?? 0),
            },
          })),
        };
        setSamplesFc(normalized);
        setLayers({ lot: true, samples: true, centroids: false, clusters: false, idw: false });
      } else {
        setSamplesFc(null);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runInspect() {
    if (!selected) return;
    setLoading(true);
    setError(null);
    setActaClosed(false);
    setCentroidsFc(null);
    setClustersFc(null);
    setIdwOverlay(null);
    try {
      const res = await api.inspect(selected.lot_id, method);
      setResult(res);
      const sid = res.session_id;

      const samples =
        (await fetchFc(api.artifactUrl(sid, 'samples.geojson'))) ||
        ((detail?.samples_geojson as FeatureCollection) ?? null);
      if (samples) setSamplesFc(samples);

      if (method !== 'traditional') {
        const [centroids, ambientes, clusters] = await Promise.all([
          fetchFc(api.artifactUrl(sid, 'centroids_damage.geojson')),
          fetchFc(api.artifactUrl(sid, 'ambientes.geojson')),
          fetchFc(api.artifactUrl(sid, 'clusters.geojson')),
        ]);
        if (centroids) setCentroidsFc(centroids);
        setClustersFc(ambientes?.features?.length ? ambientes : clusters);

        const bounds = (res.bounds || res.compare?.bounds) as number[] | undefined;
        if (bounds && bounds.length === 4) {
          setIdwOverlay({
            url: `${api.artifactUrl(sid, 'idw.png')}?t=${Date.now()}`,
            bounds: bounds as [number, number, number, number],
          });
        }
        setLayers({ lot: true, samples: true, centroids: true, clusters: true, idw: true });
      } else {
        setLayers({ lot: true, samples: true, centroids: false, clusters: false, idw: false });
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="px-5 py-4 border-b border-white/10 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.28em] text-lime-300/80">Sancor Seguros Agro</p>
          <h1 className="font-display text-3xl md:text-4xl text-white leading-none mt-1">Portal Pericial</h1>
          <p className="text-sm text-emerald-100/65 mt-1">
            Denuncia → asignación → visita y muestreo → cierre de tasación
          </p>
        </div>
        <div className="text-right text-xs text-emerald-100/60">
          <p>Tasador · Zona Marcos Juárez</p>
          <p>Plazo denuncia: 72 h · App / Portal Pericial</p>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 xl:grid-cols-[340px_1fr_360px] gap-3 p-4 min-h-0">
        <aside className="rounded-2xl border border-white/10 bg-black/25 backdrop-blur p-3 flex flex-col min-h-[320px]">
          <p className="text-[11px] uppercase tracking-widest text-lime-200 mb-2">Casos asignados</p>
          <div className="flex gap-2 mb-3">
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Buscar lote / establecimiento"
              className="flex-1 rounded-full bg-black/30 border border-white/15 px-3 py-1.5 text-sm"
            />
            <button type="button" className="btn-secondary" onClick={() => loadCases(q)}>
              Buscar
            </button>
          </div>
          <div className="overflow-auto flex-1 space-y-1.5 pr-1">
            {cases.map((c, idx) => (
              <button
                key={`${c.lot_id}-${idx}`}
                type="button"
                className={clsx('btn-ghost', selected?.lot_id === c.lot_id && 'active')}
                onClick={() => openCase(c)}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium text-white text-sm">{c.lot.name}</span>
                  <span
                    className={clsx(
                      'text-[10px] uppercase px-2 py-0.5 rounded-full',
                      c.priority === 'alta' ? 'bg-amber-400/20 text-amber-200' : 'bg-white/10 text-emerald-100/70',
                    )}
                  >
                    {c.priority}
                  </span>
                </div>
                <p className="text-[11px] text-emerald-100/55 mt-0.5">
                  {c.lot.farm_name || '—'} · {c.lot.area_ha.toFixed(0)} ha · {c.lot.sample_count} muestras
                </p>
                <p className="text-[10px] text-emerald-100/40">{c.case_id} · {c.event}</p>
              </button>
            ))}
          </div>
        </aside>

        <section className="rounded-2xl border border-white/10 bg-black/20 overflow-hidden min-h-[420px] relative">
          <MapView />
          <div className="absolute left-3 bottom-3 right-3 z-10 flex flex-wrap gap-1.5">
            {(
              [
                ['lot', 'Lote'],
                ['samples', 'Muestras'],
                ['centroids', 'Centroides'],
                ['clusters', 'Clusters'],
                ['idw', 'IDW'],
              ] as [LayerKey, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={clsx(
                  'text-[11px] rounded-full px-3 py-1 border',
                  layers[key]
                    ? 'bg-lime-400 text-black border-lime-300'
                    : 'bg-black/55 text-emerald-100/80 border-white/15',
                )}
                onClick={() => toggleLayer(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </section>

        <aside className="rounded-2xl border border-white/10 bg-black/25 backdrop-blur p-4 space-y-3 overflow-auto">
          <p className="text-[11px] uppercase tracking-widest text-lime-200">Inspección en campo</p>
          {!selected && (
            <p className="text-sm text-emerald-100/60">Seleccioná un caso: se pinnean las muestras GPS del lote.</p>
          )}
          {selected && (
            <>
              <div className="text-sm space-y-1">
                <p className="text-white font-medium">{selected.lot.name}</p>
                <p className="text-emerald-100/60 text-xs">
                  {selected.lot.crop_name || 'Cultivo —'} · {selected.lot.area_ha.toFixed(1)} ha
                </p>
                {detail && (
                  <p className="text-xs text-emerald-100/50">
                    Siniestro {String(detail.nro_siniestro || '—')} · OT {String(detail.orden_trabajo || '—')} ·{' '}
                    {String(detail.n_samples || selected.lot.sample_count)} muestras GPS
                  </p>
                )}
              </div>

              <ol className="text-[11px] space-y-1 text-emerald-100/55 border border-white/10 rounded-xl p-3">
                {(
                  (detail?.workflow as string[]) || [
                    'Denuncia ingresada',
                    'Análisis de zona y cultivo',
                    'Asignación a tasador',
                    'Visita y muestreo',
                    'Cierre de tasación',
                  ]
                ).map((step, i) => (
                  <li key={step} className={clsx(i === 3 && 'text-lime-200')}>
                    {i + 1}. {step}
                  </li>
                ))}
              </ol>

              <div>
                <p className="text-[11px] text-emerald-100/70 mb-1.5">Método de estimación</p>
                <div className="flex flex-col gap-1.5">
                  {(
                    [
                      ['traditional', 'Original · promedio simple de muestras'],
                      ['directed', 'Nuevo · K-Means + asignación + IDW + ponderado'],
                      ['both', 'Comparar ambos (+ ambientes reducidos)'],
                    ] as const
                  ).map(([id, label]) => (
                    <label key={id} className="flex items-center gap-2 text-xs cursor-pointer">
                      <input
                        type="radio"
                        name="method"
                        checked={method === id}
                        onChange={() => setMethod(id)}
                      />
                      {label}
                    </label>
                  ))}
                </div>
              </div>

              <button type="button" className="btn-primary w-full" disabled={loading} onClick={() => runInspect()}>
                {loading ? 'Procesando pipeline…' : 'Ejecutar muestreo / estimación'}
              </button>

              {result && <ResultsPanel result={result} actaClosed={actaClosed} onClose={() => setActaClosed(true)} />}
            </>
          )}

          {error && (
            <p className="text-sm text-red-200 bg-red-900/40 border border-red-400/30 rounded-xl p-3">{error}</p>
          )}
          {loading && <p className="text-xs text-lime-200 animate-pulse">Procesando…</p>}
        </aside>
      </div>
    </div>
  );
}

function ResultsPanel({
  result,
  actaClosed,
  onClose,
}: {
  result: InspectResult;
  actaClosed: boolean;
  onClose: () => void;
}) {
  const assign = (result.pipeline as { assign?: { damage_mean?: number } } | undefined)?.assign;
  return (
    <div className="space-y-2">
      <ResultCard title="Método original" value={result.traditional_mean} hint="Promedio simple de muestras GPS" />
      {result.weighted_mean != null && (
        <ResultCard
          title="Método nuevo · ponderado k"
          value={result.weighted_mean}
          hint={`K-Means k=${result.k ?? '—'} · daño asignado × área`}
        />
      )}
      {result.weighted_ambientes_mean != null && (
        <ResultCard
          title="Método nuevo · ambientes"
          value={result.weighted_ambientes_mean}
          hint={`Ambientes reducidos k'=${result.k_ambientes ?? '—'}`}
        />
      )}
      {assign?.damage_mean != null && (
        <p className="text-[11px] text-emerald-100/55">
          Daño medio en centroides (asignación NN): {Number(assign.damage_mean).toFixed(2)}%
        </p>
      )}
      {result.abs_diff != null && (
        <p className="text-[11px] text-emerald-100/55">
          Δ ponderado vs simple: {result.abs_diff.toFixed(2)} pp
          {result.abs_diff_ambientes != null &&
            ` · Δ ambientes vs simple: ${result.abs_diff_ambientes.toFixed(2)} pp`}
        </p>
      )}

      {!!result.ambientes?.length && <MiniTable title="Ambientes" rows={result.ambientes} />}
      {!result.ambientes?.length && !!result.clusters?.length && (
        <MiniTable title="Clusters" rows={result.clusters} />
      )}

      <div className="rounded-xl border border-white/15 p-3 space-y-2">
        <p className="text-[11px] uppercase tracking-wider text-lime-200/80">Acta digital de tasación</p>
        <p className="text-xs text-emerald-100/60">
          Daño estimado para liquidación:{' '}
          <strong className="text-white">
            {(
              result.weighted_ambientes_mean ??
              result.weighted_mean ??
              result.traditional_mean ??
              0
            ).toFixed(2)}
            %
          </strong>
        </p>
        <button
          type="button"
          className={clsx('w-full', actaClosed ? 'btn-secondary' : 'btn-primary')}
          onClick={onClose}
          disabled={actaClosed}
        >
          {actaClosed ? 'Acta cerrada · enviada a liquidación' : 'Cerrar tasación y recabar conformidad'}
        </button>
      </div>
    </div>
  );
}

function ResultCard({ title, value, hint }: { title: string; value?: number; hint: string }) {
  return (
    <div className="rounded-xl bg-gradient-to-br from-lime-400/15 to-amber-400/10 border border-white/12 p-3">
      <p className="text-[11px] uppercase tracking-wider text-lime-200/80">{title}</p>
      <p className="text-2xl font-semibold text-white mt-1">
        {value == null || Number.isNaN(value) ? '—' : `${value.toFixed(2)}%`}
      </p>
      <p className="text-[11px] text-emerald-100/55">{hint}</p>
    </div>
  );
}

function MiniTable({
  title,
  rows,
}: {
  title: string;
  rows: { label: string; area_ha: number; weight_pct: number; damage: number }[];
}) {
  return (
    <div className="rounded-xl border border-white/12 overflow-hidden text-[11px]">
      <div className="bg-black/30 px-2 py-1.5 text-lime-200/80 uppercase tracking-wider">{title}</div>
      <table className="w-full">
        <thead className="text-emerald-100/50">
          <tr>
            <th className="text-left px-2 py-1">ID</th>
            <th className="text-left px-2 py-1">Área</th>
            <th className="text-left px-2 py-1">Peso</th>
            <th className="text-left px-2 py-1">Daño</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-t border-white/5">
              <td className="px-2 py-1 text-white">{r.label}</td>
              <td className="px-2 py-1">{r.area_ha.toFixed(1)} ha</td>
              <td className="px-2 py-1">{r.weight_pct.toFixed(0)}%</td>
              <td className="px-2 py-1">{r.damage.toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
