import { writable } from 'svelte/store';
import type { AppSelection } from '$lib/types';

const initial: AppSelection = { gameId: 'chess', firstAgentId: 'human', secondAgentId: 'chess-best', simulations: 800, showAnalysis: true };
export const selection = writable<AppSelection>(initial);

export function persistSelection(value: AppSelection) {
  selection.set(value);
  if (typeof sessionStorage !== 'undefined') sessionStorage.setItem('engine-zoo-selection', JSON.stringify(value));
}

export function restoreSelection(): AppSelection {
  if (typeof sessionStorage === 'undefined') return initial;
  try { return { ...initial, ...JSON.parse(sessionStorage.getItem('engine-zoo-selection') ?? '{}') }; }
  catch { return initial; }
}
