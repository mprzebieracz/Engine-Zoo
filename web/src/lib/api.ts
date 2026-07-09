export type GameId = 'chess' | 'connect4';

export type LegalMove = {
  action: number;
  move: string;
};

export type SessionView = {
  id: number;
  board: string;
  moves: string[];
  san_moves: string[];
  pgn: string;
  human_turn: boolean;
  terminal: boolean;
  reward: number;
  legal_moves: LegalMove[];
};

export type MoveScore = {
  action: number;
  mv: string;
  p: number;
};

export type Analysis = {
  value: number;
  best_action: number | null;
  best_move: string | null;
  policy: MoveScore[];
};

export type AnalyzeRequest = {
  position: {
    game: 'chess';
    position: {
      fen?: string;
      moves: string[];
    };
  };
  model: string;
  mode?: 'net' | 'mcts';
  simulations: number;
  wait_for_count: number;
};

async function readJson<T>(response: Response): Promise<T> {
  const json = await response.json();
  if (!response.ok) {
    throw new Error(json.error || response.statusText);
  }
  return json as T;
}

export async function createChessSession(options: {
  engineFirst: boolean;
  model: string;
  simulations: number;
}): Promise<SessionView> {
  const response = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      position: { game: 'chess', position: { moves: [] } },
      model: options.model,
      engine_first: options.engineFirst,
      simulations: options.simulations,
      wait_for_count: 1
    })
  });
  return readJson<SessionView>(response);
}

export async function playMove(sessionId: number, move: string): Promise<SessionView> {
  const response = await fetch(`/api/sessions/${sessionId}/move`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ mv: move })
  });
  return readJson<SessionView>(response);
}

export async function analyze(request: AnalyzeRequest): Promise<Analysis> {
  const query = await fetch('/api/analyze', {
    method: 'QUERY',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(request)
  }).catch(() => null);

  if (query && query.status !== 405) {
    return readJson<Analysis>(query);
  }

  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(request)
  });
  return readJson<Analysis>(response);
}
