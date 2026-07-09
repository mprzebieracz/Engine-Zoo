export type GameId = 'chess' | 'connect4' | string;
export type AgentKind = 'human' | 'alphazero' | 'remote' | string;

export interface GameDescriptor {
  id: GameId;
  name: string;
  symbol: string;
  description: string;
  accent: string;
  available: boolean;
}

export interface AgentDescriptor {
  id: string;
  name: string;
  kind: AgentKind;
  games: GameId[];
  description?: string;
  model?: string;
  server?: string;
  badge?: string;
  defaults?: { simulations?: number; waitForCount?: number };
}

export interface AppSelection {
  gameId: string;
  firstAgentId: string;
  secondAgentId: string;
  simulations: number;
  showAnalysis: boolean;
}

export interface LegalMove { action: number; move: string }
export interface SessionView {
  id: number;
  board: string;
  moves: string[];
  san_moves: string[];
  pgn: string;
  human_turn: boolean;
  terminal: boolean;
  reward: number | null;
  legal_moves: LegalMove[];
  analysis?: Analysis;
}

export interface PolicyEntry {
  move?: string;
  mv?: string;
  display_move?: string;
  probability?: number;
  p?: number;
  prior?: number;
  visits?: number;
  visit_count?: number;
  q?: number;
  value?: number;
  logit?: number;
}

export interface Analysis {
  best_move?: string | null;
  best_move_san?: string | null;
  value?: number | null;
  policy?: PolicyEntry[] | Record<string, number>;
  network_policy?: PolicyEntry[] | Record<string, number>;
  mcts_policy?: PolicyEntry[] | Record<string, number>;
  root_policy?: PolicyEntry[] | Record<string, number>;
  moves?: PolicyEntry[];
  [key: string]: unknown;
}

export interface PuzzleCase { id: string; name: string; category: string; position: unknown; expected: string[] }
export interface PuzzleResult extends PuzzleCase {
  status: 'pending' | 'running' | 'passed' | 'failed' | 'error';
  bestMove?: string;
  analysis?: Analysis;
  error?: string;
}
