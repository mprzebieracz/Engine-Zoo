import type { AgentDescriptor, GameDescriptor, PuzzleCase } from '$lib/types';

export const games: GameDescriptor[] = [
  {
    id: 'chess',
    name: 'Chess',
    symbol: '♞',
    description: 'Classical chess with legal move highlighting and engine inspection.',
    accent: '#b7f36b',
    available: true
  },
  {
    id: 'connect4',
    name: 'Connect Four',
    symbol: '●',
    description: 'Fast tactical games on a seven-column board.',
    accent: '#70a7ff',
    available: true
  },
  {
    id: 'go',
    name: 'Go',
    symbol: '◉',
    description: 'Reserved for a future game adapter.',
    accent: '#f3b96b',
    available: false
  }
];

// This is a frontend development fallback. In production, fetch this from
// GET /api/catalog so model paths and credentials remain server-side.
export const agents: AgentDescriptor[] = [
  {
    id: 'human',
    name: 'Human',
    kind: 'human',
    games: ['chess', 'connect4'],
    description: 'Moves are entered through the board UI.'
  },
  {
    id: 'chess-best',
    name: 'AlphaZero · best',
    kind: 'alphazero',
    games: ['chess'],
    model: 'best',
    description: 'Current promoted checkpoint.'
  },
  {
    id: 'chess-candidate',
    name: 'AlphaZero · candidate',
    kind: 'alphazero',
    games: ['chess'],
    model: 'candidate',
    description: 'Latest candidate checkpoint.'
  },
  {
    id: 'connect4-best',
    name: 'AlphaZero · best',
    kind: 'alphazero',
    games: ['connect4'],
    model: 'best',
    description: 'Current promoted checkpoint.'
  }
];

export const samplePuzzles: PuzzleCase[] = [
  {
    id: 'chess-mate-1',
    name: 'Back-rank finish',
    category: 'Mate in one',
    position: { game: 'chess', position: { fen: '6k1/5ppp/8/8/8/8/5PPP/6RK w - - 0 1', moves: [] } },
    expected: ['g1e8']
  },
  {
    id: 'chess-opening-1',
    name: 'Starting position',
    category: 'Policy sanity',
    position: { game: 'chess', position: { fen: null, moves: [] } },
    expected: ['e2e4', 'd2d4', 'g1f3', 'c2c4']
  }
];
