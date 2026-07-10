import { Chess } from 'chess.js';
import type { GameId, LegalMove } from '$lib/types';

export interface MatchPosition {
  legalMoves: LegalMove[];
  sanMoves: string[];
  terminal: boolean;
  result: string;
  turn: 0 | 1;
}

export function inspectPosition(gameId: GameId, moves: string[]): MatchPosition {
  return gameId === 'chess' ? inspectChess(moves) : inspectConnect4(moves);
}

export function positionPayload(gameId: GameId, moves: string[]) {
  return gameId === 'chess'
    ? { game: 'chess', position: { fen: null, moves } }
    : { game: 'connect4', position: { moves: moves.map(Number) } };
}

export function lastMoveSquares(gameId: GameId, moves: string[]): [string, string] | undefined {
  const move = moves.at(-1);
  return gameId === 'chess' && move ? [move.slice(0, 2), move.slice(2, 4)] : undefined;
}

function inspectChess(moves: string[]): MatchPosition {
  const chess = new Chess();
  const sanMoves: string[] = [];
  for (const uci of moves) {
    const played = chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' });
    if (played) sanMoves.push(played.san);
  }
  const legalMoves = chess.moves({ verbose: true }).map((move, action) => ({
    action,
    move: `${move.from}${move.to}${move.promotion ?? ''}`
  }));
  const result = chess.isCheckmate()
    ? `${chess.turn() === 'w' ? 'Black' : 'White'} wins by checkmate`
    : chess.isDraw() ? 'Draw' : '';
  return { legalMoves, sanMoves, terminal: chess.isGameOver(), result, turn: chess.turn() === 'w' ? 0 : 1 };
}

function inspectConnect4(moves: string[]): MatchPosition {
  const board = Array.from({ length: 6 }, () => Array<number>(7).fill(-1));
  let winner = -1;
  for (let ply = 0; ply < moves.length; ply += 1) {
    const col = Number(moves[ply]);
    const row = board.findLastIndex((line) => line[col] === -1);
    if (row >= 0) board[row][col] = ply % 2;
    if (row >= 0 && hasFour(board, row, col, ply % 2)) winner = ply % 2;
  }
  const legalMoves = winner < 0
    ? board[0].flatMap((cell, col) => cell === -1 ? [{ action: col, move: String(col) }] : [])
    : [];
  const full = moves.length >= 42;
  return {
    legalMoves,
    sanMoves: [...moves],
    terminal: winner >= 0 || full,
    result: winner >= 0 ? `${winner === 0 ? 'First' : 'Second'} wins` : full ? 'Draw' : '',
    turn: moves.length % 2 as 0 | 1
  };
}

function hasFour(board: number[][], row: number, col: number, player: number): boolean {
  return [[0, 1], [1, 0], [1, 1], [1, -1]].some(([dr, dc]) => {
    let count = 1;
    for (const direction of [-1, 1]) {
      for (let distance = 1; distance < 4; distance += 1) {
        const r = row + dr * distance * direction;
        const c = col + dc * distance * direction;
        if (board[r]?.[c] !== player) break;
        count += 1;
      }
    }
    return count >= 4;
  });
}
