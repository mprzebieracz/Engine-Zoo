import type { Analysis, SessionView } from '$lib/types';

const ENV_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? '';

function base(server?: string) {
  return server?.replace(/\/$/, '') || ENV_BASE;
}

async function responseBody(response: Response): Promise<unknown> {
  const body = await response.text();

  if (!body) return undefined;

  try {
    return JSON.parse(body);
  } catch {
    return body;
  }
}

function responseError(response: Response, body: unknown): Error {
  const message = typeof body === 'object' && body && 'error' in body
    ? String((body as { error: unknown }).error)
    : `${response.status} ${response.statusText}`;

  return new Error(message);
}

export async function request<T>(path: string, init?: RequestInit, server?: string): Promise<T> {
  const response = await fetch(`${base(server)}${path}`, {
    ...init,
    headers: { 'content-type': 'application/json', ...init?.headers }
  });

  const parsed = await responseBody(response);

  if (!response.ok) throw responseError(response, parsed);

  return parsed as T;
}

export const api = {
  health: (server?: string) => request<{ status: string }>('/api/health', undefined, server),
  createSession: (payload: { position?: unknown; model: string; engine_first: boolean; simulations: number; wait_for_count: number }, server?: string) =>
    request<SessionView>('/api/sessions', { method: 'POST', body: JSON.stringify(payload) }, server),
  move: (id: number, mv: string, server?: string) =>
    request<SessionView>(`/api/sessions/${id}/move`, { method: 'POST', body: JSON.stringify({ mv }) }, server),
  engineMove: (id: number, server?: string) =>
    request<SessionView>(`/api/sessions/${id}/engine`, { method: 'POST' }, server),
  analyze: (payload: { position: unknown; model: string; mode: 'net' | 'mcts'; simulations: number; wait_for_count: number }, server?: string) =>
    request<Analysis>('/api/analyze', { method: 'POST', body: JSON.stringify(payload) }, server)
};
