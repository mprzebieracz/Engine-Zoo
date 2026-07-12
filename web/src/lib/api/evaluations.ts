import { request } from '$lib/api/client';
import type { Catalog, EvaluationJob, Suite } from '$lib/evaluations/types';

export const evaluationsApi = {
  catalog: (server?: string) => request<Catalog>('/api/evaluations/catalog', undefined, server),
  jobs: (server?: string) => request<EvaluationJob[]>('/api/evaluations/jobs', undefined, server),
  job: (id: string, server?: string) => request<EvaluationJob>(`/api/evaluations/jobs/${encodeURIComponent(id)}`, undefined, server),
  create: (payload: { candidate: string; suite: Suite; baseline?: string; simulations: number; rounds: number; stockfish_nodes: number }, server?: string) =>
    request<EvaluationJob>('/api/evaluations/jobs', { method: 'POST', body: JSON.stringify(payload) }, server),
  artifactUrl: (id: string, name: string) => `/api/evaluations/jobs/${encodeURIComponent(id)}/artifacts/${encodeURIComponent(name)}`
};
