<script lang="ts">
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';
  import { Chess } from 'chess.js';
  import { api } from '$lib/api/client';
  import { loadAgents } from '$lib/config/agents';
  import { games } from '$lib/config/games';
  import { restoreSelection } from '$lib/state/selection';
  import ChessgroundBoard from '$lib/components/ChessgroundBoard.svelte';
  import Connect4Board from '$lib/components/Connect4Board.svelte';
  import AnalysisPanel from '$lib/components/AnalysisPanel.svelte';
  import PlayerStrip from '$lib/components/PlayerStrip.svelte';
  import type { AgentDescriptor, Analysis, AppSelection, PolicyEntry, SessionView } from '$lib/types';

  let config = $state<AppSelection>(restoreSelection());
  let agents = $state<AgentDescriptor[]>([]);
  let session = $state<SessionView | null>(null);
  let analysis = $state<Analysis | undefined>();
  let loading = $state(true);
  let engineThinking = $state(false);
  let error = $state('');

  const game = $derived(games.find((entry) => entry.id === config.gameId));
  const first = $derived(agents.find((agent) => agent.id === config.firstAgentId));
  const second = $derived(agents.find((agent) => agent.id === config.secondAgentId));
  const engine = $derived(first?.kind === 'human' ? second : first);
  const humanFirst = $derived(first?.kind === 'human');
  const engineSimulations = $derived(engine?.defaults?.simulations ?? config.simulations);
  const engineWaitForCount = $derived(engine?.defaults?.waitForCount ?? 1);
  const firstActive = $derived(turnActive(first));
  const secondActive = $derived(turnActive(second));

  onMount(async () => {
    config = restoreSelection();
    agents = await loadAgents();
    if (!agents.find((agent) => agent.id === config.firstAgentId) || !agents.find((agent) => agent.id === config.secondAgentId)) {
      goto('/setup'); return;
    }
    await startGame();
  });

  async function startGame() {
    loading = true; error = ''; analysis = undefined;
    try {
      session = await api.createSession({ model: engine?.model ?? 'best', engine_first: !humanFirst, simulations: engineSimulations, wait_for_count: engineWaitForCount }, engine?.server);
      if (session.terminal) return;
      if (session.human_turn) {
        if (config.showAnalysis) await analyzeCurrent();
      } else {
        await runEngineTurn();
      }
    } catch (cause) { error = cause instanceof Error ? cause.message : String(cause); }
    finally { loading = false; }
  }

  function turnActive(agent?: AgentDescriptor): boolean {
    if (!session || !agent) return false;
    return agent.kind === 'human' ? session.human_turn : !session.human_turn;
  }

  function position() {
    return positionFromMoves(session?.moves ?? []);
  }

  function positionFromMoves(moves: string[]) {
    return config.gameId === 'chess'
      ? { game: 'chess', position: { fen: null, moves } }
      : { game: 'connect4', position: { moves: moves.map(Number) } };
  }

  function sanFor(moves: string[], uci?: string | null): string | null {
    if (config.gameId !== 'chess' || !uci) return uci ?? null;
    try {
      const chess = new Chess();
      for (const move of moves) chess.move({ from: move.slice(0, 2), to: move.slice(2, 4), promotion: move[4] || 'q' });
      return chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' })?.san ?? uci;
    } catch {
      return uci;
    }
  }

  function decorateAnalysis(result: Analysis, moves: string[]): Analysis {
    if (config.gameId !== 'chess') return result;
    const decorateRow = (row: PolicyEntry): PolicyEntry => {
      const entry = row as { move?: string; mv?: string };
      const uci = entry.move ?? entry.mv;
      return { ...entry, display_move: sanFor(moves, uci) ?? uci };
    };
    return {
      ...result,
      best_move_san: sanFor(moves, result.best_move),
      policy: Array.isArray(result.policy) ? result.policy.map(decorateRow) : result.policy,
      network_policy: Array.isArray(result.network_policy) ? result.network_policy.map(decorateRow) : result.network_policy,
      mcts_policy: Array.isArray(result.mcts_policy) ? result.mcts_policy.map(decorateRow) : result.mcts_policy,
      root_policy: Array.isArray(result.root_policy) ? result.root_policy.map(decorateRow) : result.root_policy,
      moves: Array.isArray(result.moves) ? result.moves.map(decorateRow) : result.moves
    };
  }

  async function analyzeCurrent() {
    if (!session || !engine) return;
    loading = true;
    try {
      const moves = [...session.moves];
      const result = await api.analyze({ position: positionFromMoves(moves), model: engine.model ?? 'best', mode: 'mcts', simulations: engineSimulations, wait_for_count: engineWaitForCount }, engine.server);
      analysis = decorateAnalysis(result, moves);
    } catch (cause) { error = cause instanceof Error ? cause.message : String(cause); }
    finally { loading = false; }
  }

  async function makeMove(move:string) {
    if (!session || !session.human_turn || session.terminal || !engine) return;
    loading = true; error = '';
    try {
      session = await api.move(session.id, move, engine.server);
      loading = false;
      if (session.terminal) return;

      await runEngineTurn();
    } catch (cause) { error = cause instanceof Error ? cause.message : String(cause); }
    finally { loading = false; engineThinking = false; }
  }

  async function runEngineTurn() {
    if (!session || session.human_turn || session.terminal || !engine) return;
    if (config.showAnalysis) await analyzeCurrent();
    engineThinking = true;
    session = await api.engineMove(session.id, engine.server);
    engineThinking = false;
  }
</script>
<svelte:head><title>{game?.name ?? 'Match'} · Engine Zoo</title></svelte:head>
<main class="page screen-enter play-page">
  <header class="match-head">
    <div><button onclick={()=>goto('/setup')}>← Match setup</button><span class="eyebrow">Step 3 of 3</span><h1>{game?.name}</h1></div>
    <div class="head-actions"><span class:thinking={loading||engineThinking} class="thinking-pill">{engineThinking?'Engine thinking':loading?'Loading':'Live'}</span><button class="secondary" onclick={startGame}>New game</button></div>
  </header>
  {#if error}<div class="error">{error}</div>{/if}
  {#if !session}
    <div class="loading-board"><span></span><p>Creating match…</p></div>
  {:else}
    <section class="arena">
      <div class="board-column">
        <PlayerStrip agent={second} side={config.gameId==='chess'?'Black':'Second'} active={secondActive} />
        {#if config.gameId==='chess'}
          <ChessgroundBoard moves={session.moves} legalMoves={session.legal_moves} disabled={!session.human_turn||session.terminal||loading||engineThinking} orientation={humanFirst?'white':'black'} onmove={makeMove}/>
        {:else}
          <Connect4Board moves={session.moves} disabled={!session.human_turn||session.terminal||loading||engineThinking} onmove={makeMove}/>
        {/if}
        <PlayerStrip agent={first} side={config.gameId==='chess'?'White':'First'} active={firstActive} />
      </div>
      <aside>
        <section class="moves"><header><h2>Move history</h2><span>{session.moves.length} plies</span></header><div class="move-list">{#each (session.san_moves.length?session.san_moves:session.moves) as move,i}<span><small>{i+1}</small>{move}</span>{/each}</div></section>
        {#if config.showAnalysis}<AnalysisPanel {analysis} {loading}/>{/if}
        <button class="secondary analyze" onclick={analyzeCurrent} disabled={loading||engineThinking}>Analyze current position</button>
      </aside>
    </section>
  {/if}
</main>

<style>
  .play-page{padding-top:34px}.match-head{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:24px}.match-head button:first-child{display:block;border:0;background:transparent;color:var(--muted);padding:0;margin-bottom:17px;cursor:pointer}.match-head h1{font-size:34px;margin:5px 0 0}.head-actions{display:flex;align-items:center;gap:10px}.thinking-pill{padding:7px 10px;border-radius:99px;background:color-mix(in srgb,var(--accent) 10%,transparent);color:var(--accent);font-size:10px;text-transform:uppercase;letter-spacing:.08em}.thinking{animation:pulse 1s infinite alternate}.arena{display:grid;grid-template-columns:minmax(0,820px) minmax(330px,1fr);gap:30px;align-items:start}.board-column{display:flex;flex-direction:column;align-items:center;gap:10px}.player-strip{width:min(74vh,100%);display:grid;grid-template-columns:38px 1fr 8px auto;gap:10px;align-items:center;padding:8px 10px;border-radius:12px;color:var(--muted);transition:.2s}.player-strip.active{background:var(--panel)}.avatar{width:34px;height:34px;border-radius:9px;display:grid;place-items:center;background:var(--panel-2);color:var(--accent);font-weight:800}.player-strip strong,.player-strip small{display:block}.player-strip strong{color:var(--text);font-size:12px}.player-strip small,.player-strip span{font-size:9px;text-transform:uppercase;letter-spacing:.08em}.player-strip i{width:7px;height:7px;border-radius:50%;background:#414752}.player-strip.active i{background:var(--accent);box-shadow:0 0 10px var(--accent)}aside{display:grid;gap:15px;position:sticky;top:98px}.moves{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:18px}.moves header{display:flex;justify-content:space-between;align-items:center}.moves h2{font-size:16px;margin:0}.moves header span{font-size:10px;color:var(--muted)}.move-list{max-height:160px;overflow:auto;display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:13px}.move-list>span{background:var(--panel-2);padding:7px 8px;border-radius:7px;font:11px ui-monospace,monospace}.move-list small{color:var(--muted);margin-right:8px}.analyze{width:100%}.loading-board{height:520px;display:grid;place-items:center;align-content:center;color:var(--muted)}.loading-board span{width:35px;height:35px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}.error{margin-bottom:18px}@keyframes spin{to{transform:rotate(360deg)}}@keyframes pulse{to{opacity:.42}}@media(max-width:980px){.arena{grid-template-columns:1fr}aside{position:static}.match-head{align-items:flex-start}.head-actions{margin-top:10px}}@media(max-width:600px){.match-head{display:block}.head-actions{justify-content:space-between}.player-strip{width:100%}}
</style>
