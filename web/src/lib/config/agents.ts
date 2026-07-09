import type { AgentDescriptor } from '$lib/types';

const fallback: AgentDescriptor[] = [
  { id: 'human', name: 'Human', kind: 'human', games: ['chess', 'connect4'], description: 'Moves are entered through the board.' },
  { id: 'chess-best', name: 'AlphaZero · best', kind: 'alphazero', games: ['chess'], model: 'best', server: 'http://127.0.0.1:8080' },
  { id: 'connect4-best', name: 'AlphaZero · best', kind: 'alphazero', games: ['connect4'], model: 'best', server: 'http://127.0.0.1:8081' }
];

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
  return agents.length ? agents : fallback;
}

export async function loadAgents(): Promise<AgentDescriptor[]> {
  try {
    const response = await fetch('/config/agents.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status}`);
    return normalizeAgents(await response.json());
  } catch {
    return fallback;
  }
}
