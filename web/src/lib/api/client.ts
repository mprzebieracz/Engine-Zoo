import type { Analysis, SessionView } from '$lib/types';

const ENV_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? '';

function base(server?: string) {
  return (server?.replace(/\/$/, '') || ENV_BASE);
}

export async function request<T>(path: string, init?: RequestInit, server?: string): Promise<T> {
  const response = await fetch(`${base(server)}${path}`, {
    ...init,
    headers: { 'content-type': 'application/json', ...init?.headers }
  });
  const body = await response.text();
  let parsed: unknown = undefined;
  if (body) { try { parsed = JSON.parse(body); } catch { parsed = body; } }
  if (!response.ok) {
    const message = typeof parsed === 'object' && parsed && 'error' in parsed
      ? String((parsed as { error: unknown }).error)
      : `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
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
