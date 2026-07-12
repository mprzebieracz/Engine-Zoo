export type Suite = 'puzzle' | 'stockfish' | 'arena';
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'interrupted';

export interface PreflightState { available: boolean; message: string }
export interface Catalog {
  checkpoints: string[];
  suites: { id: Suite; label: string; defaults: { simulations?: number; rounds?: number; stockfish_nodes?: number } }[];
  preflight: Record<Suite, PreflightState>;
}
export interface EvaluationJob {
  id: string; candidate: string; suite: Suite; baseline?: string; simulations: number; rounds: number;
  stockfish_nodes: number; status: JobStatus; created_at: number; updated_at: number; error?: string;
  artifacts: string[]; result?: Record<string, unknown>;
}
