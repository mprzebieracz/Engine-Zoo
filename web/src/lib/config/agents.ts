import type { AgentDescriptor } from '$lib/types';

function isAgent(value: unknown): value is AgentDescriptor {
  if (!value || typeof value !== 'object') return false;
  const agent = value as Partial<AgentDescriptor>;
  return typeof agent.id === 'string'
    && typeof agent.name === 'string'
    && typeof agent.kind === 'string'
    && Array.isArray(agent.games)
    && agent.games.every((game) => typeof game === 'string');
}

function normalizeAgents(value: unknown): AgentDescriptor[] {
  const agents = Array.isArray((value as { agents?: unknown })?.agents)
    ? (value as { agents: unknown[] }).agents.filter(isAgent)
    : [];
  if (!agents.length) throw new Error('No valid agents in /config/agents.json');
  return agents;
}

export async function loadAgents(): Promise<AgentDescriptor[]> {
  const response = await fetch(`/config/agents.json?t=${Date.now()}`, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load /config/agents.json (${response.status})`);
  return normalizeAgents(await response.json());
}
