import { Chess } from 'chess.js';
import type { AgentDescriptor, Analysis, PlayerSelection, PolicyEntry } from '$lib/types';

export function analysisKey(side: number, player: PlayerSelection, moves: string[]): string {
  const behavior = player.behavior;
  return [side, player.agentId, behavior.simulations, behavior.waitForCount, behavior.movePolicy, behavior.temperature, moves.join(',')].join('|');
}

export function decorateAnalysis(result: Analysis, moves: string[], chess: boolean): Analysis {
  if (!chess) return result;
  const decorateRows = (value: unknown) => Array.isArray(value)
    ? value.map((row) => decorateRow(row as PolicyEntry, moves))
    : value;
  return {
    ...result,
    best_move_san: sanFor(moves, result.best_move),
    policy: decorateRows(result.policy) as Analysis['policy'],
    network_policy: decorateRows(result.network_policy) as Analysis['network_policy'],
    mcts_policy: decorateRows(result.mcts_policy) as Analysis['mcts_policy']
  };
}

export function chooseMove(analysis: Analysis, player: PlayerSelection): string | null {
  if (player.behavior.movePolicy === 'strict') return analysis.best_move ?? null;
  const rows = policyRows(analysis.mcts_policy ?? analysis.policy);
  if (!rows.length) return analysis.best_move ?? null;
  const temperature = Math.max(0.05, player.behavior.temperature);
  const weights = rows.map((row) => Math.pow(Math.max(0, probability(row)), 1 / temperature));
  let cursor = Math.random() * weights.reduce((sum, weight) => sum + weight, 0);
  for (let index = 0; index < rows.length; index += 1) {
    cursor -= weights[index];
    if (cursor <= 0) return rows[index].move ?? rows[index].mv ?? analysis.best_move ?? null;
  }
  return analysis.best_move ?? null;
}

export function agentLabel(agent: AgentDescriptor | undefined): string {
  return agent?.name ?? 'Agent';
}

function policyRows(value: unknown): PolicyEntry[] {
  if (Array.isArray(value)) return value as PolicyEntry[];
  if (value && typeof value === 'object') {
    return Object.entries(value as Record<string, number>).map(([move, p]) => ({ move, p }));
  }
  return [];
}

function probability(row: PolicyEntry): number {
  return row.p ?? row.probability ?? row.prior ?? 0;
}

function decorateRow(row: PolicyEntry, moves: string[]): PolicyEntry {
  const uci = row.move ?? row.mv;
  return { ...row, display_move: sanFor(moves, uci) ?? uci };
}

function sanFor(moves: string[], uci?: string | null): string | null {
  if (!uci) return uci ?? null;
  try {
    const chess = new Chess();
    for (const move of moves) chess.move({ from: move.slice(0, 2), to: move.slice(2, 4), promotion: move[4] || 'q' });
    return chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' })?.san ?? uci;
  } catch {
    return uci;
  }
}
