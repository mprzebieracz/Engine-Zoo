import type { AppSelection } from '$lib/types';

const behavior = () => ({ simulations: 800, waitForCount: 1, movePolicy: 'strict' as const, temperature: 1 });
const initial = (): AppSelection => ({
  gameId: 'chess',
  first: { agentId: '', behavior: behavior() },
  second: { agentId: '', behavior: behavior() }
});

let current = initial();

export function persistSelection(value: AppSelection) {
  current = structuredClone(value);
}

export function restoreSelection(): AppSelection {
  return structuredClone(current);
}
