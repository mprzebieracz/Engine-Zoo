import type { GameDescriptor } from '$lib/types';

export const games: GameDescriptor[] = [
  { id: 'chess', name: 'Chess', symbol: '♞', description: 'Classical chess with full engine inspection.', accent: '#b7f36b', available: true },
  { id: 'connect4', name: 'Connect Four', symbol: '●', description: 'Fast tactical play on a seven-column board.', accent: '#70a7ff', available: true },
  { id: 'go', name: 'Go', symbol: '◉', description: 'Ready for a future game adapter.', accent: '#f3b96b', available: false }
];
