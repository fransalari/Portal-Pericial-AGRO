export type LotSummary = {
  lot_id: string;
  name: string;
  farm_name?: string | null;
  crop_name?: string | null;
  area_ha: number;
  sample_count: number;
  bbox?: number[];
  campaign?: string | null;
};

export type CaseItem = {
  case_id: string;
  lot_id: string;
  status: string;
  priority: string;
  event: string;
  assignee: string;
  sla_hours: number;
  lot: LotSummary;
};

export type ClusterRow = {
  cluster: number;
  label: string;
  n_cells: number;
  area_ha: number;
  weight_pct: number;
  damage: number;
  contribution: number;
  contribution_pct: number;
};

export type InspectResult = {
  session_id: string;
  method: string;
  traditional_mean?: number;
  weighted_mean?: number;
  weighted_ambientes_mean?: number;
  directed_mean?: number;
  abs_diff?: number;
  abs_diff_ambientes?: number;
  k?: number;
  k_ambientes?: number;
  clusters?: ClusterRow[];
  ambientes?: ClusterRow[];
  n_samples?: number;
  bounds?: number[];
  compare?: Record<string, unknown> & { bounds?: number[] };
  pipeline?: Record<string, unknown>;
  samples_url?: string;
  centroids_url?: string;
  clusters_url?: string;
  ambientes_url?: string;
  idw_png_url?: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ ok: boolean; index_ready: boolean }>('/api/health'),
  cases: (q?: string) =>
    request<{ count: number; cases: CaseItem[] }>(
      `/api/cases${q ? `?q=${encodeURIComponent(q)}` : ''}`,
    ),
  caseDetail: (lotId: string) => request<Record<string, unknown>>(`/api/cases/${lotId}`),
  inspect: (lotId: string, method: 'traditional' | 'directed' | 'both' = 'both') =>
    request<InspectResult>(`/api/cases/${lotId}/inspect`, {
      method: 'POST',
      body: JSON.stringify({ grid_res_m: 10, method }),
    }),
  artifactUrl: (sessionId: string, name: string) =>
    `/api/sessions/${sessionId}/artifacts/${name}`,
};
