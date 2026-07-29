<script lang="ts">
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { onDestroy, onMount } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';
  import AnalysisPanel from '$lib/components/AnalysisPanel.svelte';
  import ChessgroundBoard from '$lib/components/ChessgroundBoard.svelte';
  import Connect4Board from '$lib/components/Connect4Board.svelte';
  import MatchControls from '$lib/components/MatchControls.svelte';
  import PlayerStrip from '$lib/components/PlayerStrip.svelte';
  import { api } from '$lib/api/client';
  import { loadAgents } from '$lib/config/agents';
  import { games } from '$lib/config/games';
  import { analysisKey, chooseMove, decorateAnalysis } from '$lib/match/analysis';
  import { inspectPosition, positionPayload } from '$lib/match/game';
  import { restoreSelection } from '$lib/state/selection';
  import type { AgentDescriptor, Analysis, AppSelection, MatchSnapshot } from '$lib/types';

  let config = $state<AppSelection>(restoreSelection());
  let agents = $state<AgentDescriptor[]>([]);
  let moves = $state<string[]>([]);
  let viewPly = $state(0);
  let snapshots = $state<MatchSnapshot[]>([]);
  let thinkingSide = $state<0 | 1 | null>(null);
  let playing = $state(false);
  let pace = $state(700);
  let loading = $state(true);
  let error = $state('');
  let boardOrientation = $state<'white' | 'black'>('white');
  let timer: ReturnType<typeof setTimeout> | undefined;
  const cache = new SvelteMap<string, Analysis>();

  const game = $derived(games.find((entry) => entry.id === config.gameId));
  const first = $derived(agents.find((agent) => agent.id === config.first.agentId));
  const second = $derived(agents.find((agent) => agent.id === config.second.agentId));
  const livePosition = $derived(inspectPosition(config.gameId, moves));
  const visibleMoves = $derived(moves.slice(0, viewPly));
  const visiblePosition = $derived(inspectPosition(config.gameId, visibleMoves));
  const automated = $derived(first?.kind !== 'human' && second?.kind !== 'human');
  const boardDisabled = $derived(viewPly !== moves.length || visiblePosition.terminal || loading || thinkingSide !== null || agentAt(livePosition.turn)?.kind !== 'human');
  const topSide = $derived((boardOrientation === 'white' ? 1 : 0) as 0 | 1);
  const bottomSide = $derived((topSide === 0 ? 1 : 0) as 0 | 1);

  onMount(() => {
    document.body.classList.add('match-view');
    void initialize();
    return () => document.body.classList.remove('match-view');
  });
  onDestroy(() => clearTimeout(timer));

  async function initialize() {
    loading = true;

    try {
      agents = await loadAgents();

      if (!agentAt(0) || !agentAt(1)) {
        await goto(resolve('/setup'));
        return;
      }

      boardOrientation = first?.kind === 'human' ? 'white' : second?.kind === 'human' ? 'black' : 'white';
      resetMatch();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      loading = false;
      void advanceIfNeeded();
    }
  }

  function playerAt(side: 0 | 1) {
    return side === 0 ? config.first : config.second;
  }

  function agentAt(side: 0 | 1) {
    return side === 0 ? first : second;
  }

  function sideLabel(side: 0 | 1) {
    return config.gameId === 'chess' ? (side === 0 ? 'White' : 'Black') : (side === 0 ? 'First' : 'Second');
  }

  function toggleOrientation() {
    boardOrientation = boardOrientation === 'white' ? 'black' : 'white';
  }

  function resetMatch() {
    clearTimeout(timer);
    moves = [];
    viewPly = 0;
    snapshots = [];
    error = '';
    playing = automated;
  }

  function cachedOutput(side: 0 | 1): Analysis | undefined {
    return [...snapshots].reverse().find((snapshot) => snapshot.side === side && snapshot.ply <= viewPly)?.analysis;
  }

  async function analyzeAgent(side: 0 | 1, positionMoves: string[]): Promise<Analysis> {
    const player = playerAt(side);
    const agent = agentAt(side);

    if (!agent || agent.kind === 'human') throw new Error('A human agent cannot be analyzed.');

    const key = analysisKey(side, player, positionMoves);
    const existing = cache.get(key);

    if (existing) return existing;

    const result = await api.analyze({
      position: positionPayload(config.gameId, positionMoves),
      model: agent.model ?? 'best',
      mode: 'mcts',
      simulations: player.behavior.simulations,
      wait_for_count: player.behavior.waitForCount
    }, agent.server);

    const decorated = decorateAnalysis(result, positionMoves, config.gameId === 'chess');
    cache.set(key, decorated);

    return decorated;
  }

  async function advanceIfNeeded() {
    clearTimeout(timer);

    if (thinkingSide !== null || livePosition.terminal) return;

    const side = livePosition.turn;
    const agent = agentAt(side);

    if (!agent || agent.kind === 'human' || (automated && !playing)) return;

    const positionMoves = [...moves];
    const followedLive = viewPly === moves.length;

    thinkingSide = side;
    error = '';

    try {
      const analysis = await analyzeAgent(side, positionMoves);

      recordAnalysis(side, agent.id, positionMoves.length, analysis);

      const move = chooseMove(analysis, playerAt(side));
      if (!move || !livePosition.legalMoves.some((entry) => entry.move === move)) {
        throw new Error(`${agent.name} returned no legal move.`);
      }

      moves = [...moves, move];

      if (followedLive) viewPly = moves.length;
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
      playing = false;
    } finally {
      thinkingSide = null;
    }

    scheduleNextMove();
  }

  function recordAnalysis(side: 0 | 1, agentId: string, ply: number, analysis: Analysis) {
    const isCurrentSnapshot = (snapshot: MatchSnapshot) => snapshot.ply === ply && snapshot.side === side;
    const snapshot = { ply, side, agentId, analysis };

    snapshots = [...snapshots.filter((entry) => !isCurrentSnapshot(entry)), snapshot];
  }

  function scheduleNextMove() {
    const position = inspectPosition(config.gameId, moves);

    if (error || position.terminal) return;

    if (automated && playing) {
      timer = setTimeout(() => void advanceIfNeeded(), pace);
      return;
    }

    if (agentAt(position.turn)?.kind !== 'human') void advanceIfNeeded();
  }

  function makeMove(move: string) {
    if (boardDisabled || !livePosition.legalMoves.some((entry) => entry.move === move)) return;
    moves = [...moves, move];
    viewPly = moves.length;
    void advanceIfNeeded();
  }

  function togglePlayback() {
    playing = !playing;
    if (playing) void advanceIfNeeded();
    else clearTimeout(timer);
  }

  function updatePace(value: number) {
    pace = value;
    if (automated && playing && thinkingSide === null) {
      clearTimeout(timer);
      timer = setTimeout(() => void advanceIfNeeded(), pace);
    }
  }
</script>

<svelte:head><title>{game?.name ?? 'Match'} · Engine Zoo</title></svelte:head>

<main class="play-page">
  <header class="match-head">
    <div class="match-title"><button class="back-button" onclick={() => goto(resolve('/setup'))}>← Setup</button><h1>{game?.name}</h1>{#if livePosition.result}<span>{livePosition.result}</span>{/if}</div>
    <div class="head-actions"><span class:thinking={thinkingSide !== null} class="status">{thinkingSide !== null ? `${agentAt(thinkingSide)?.name} thinking` : playing ? 'Playing' : 'Paused'}</span><button class="secondary" disabled={thinkingSide !== null} onclick={() => { resetMatch(); void advanceIfNeeded(); }}>New match</button></div>
  </header>

  {#if error}<div class="error">{error}</div>{/if}
  {#if loading}
    <div class="loading-board"><span></span><p>Loading agents…</p></div>
  {:else}
    <section class="arena">
      <div class="board-column">
        <PlayerStrip agent={agentAt(topSide)} side={sideLabel(topSide)} active={visiblePosition.turn === topSide && !visiblePosition.terminal} />
        <div class="board-tools"><button class="board-flip" onclick={toggleOrientation} aria-label="Flip board" title="Flip board">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 3v14m0 0-4-4m4 4 4-4M17 21V7m0 0-4 4m4-4 4 4" /></svg>
        </button></div>
        <div class="board-frame">
          {#if config.gameId === 'chess'}
            <ChessgroundBoard moves={visibleMoves} legalMoves={visiblePosition.legalMoves} disabled={boardDisabled} orientation={boardOrientation} onmove={makeMove} />
          {:else}
            <Connect4Board moves={visibleMoves} disabled={boardDisabled} onmove={makeMove} />
          {/if}
        </div>
        <PlayerStrip agent={agentAt(bottomSide)} side={sideLabel(bottomSide)} active={visiblePosition.turn === bottomSide && !visiblePosition.terminal} />
        <MatchControls ply={viewPly} total={moves.length} autoplay={playing} {pace} {automated} terminal={livePosition.terminal} onprevious={() => viewPly = Math.max(0, viewPly - 1)} onnext={() => viewPly = Math.min(moves.length, viewPly + 1)} onlive={() => viewPly = moves.length} ontoggle={togglePlayback} onpace={updatePace} />
      </div>

      <aside class="outputs">
        {#each [topSide, bottomSide] as side (side)}
          {@const sideAgent = agentAt(side)}
          {#if sideAgent?.kind !== 'human'}<AnalysisPanel title={sideAgent?.name} subtitle={sideLabel(side)} analysis={cachedOutput(side)} loading={thinkingSide === side} />{/if}
        {/each}
        {#if first?.kind === 'human' && second?.kind === 'human'}<div class="human-match"><strong>Human match</strong><p>Use the board controls to play both sides. Position navigation remains available throughout the game.</p></div>{/if}
      </aside>
    </section>
  {/if}
</main>

<style>
  :global(body.match-view){overflow:hidden}.play-page{height:calc(100dvh - 64px);max-width:1440px;margin:auto;padding:14px 24px 16px;overflow:hidden;display:flex;flex-direction:column}.match-head{height:46px;display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:10px}.match-title,.head-actions{display:flex;align-items:center;gap:12px}.match-title button{border:0;background:transparent;color:var(--muted);padding:0;cursor:pointer}.match-title h1{font-size:22px;margin:0}.match-title>span{color:var(--accent);font-size:11px}.status{padding:6px 9px;border-radius:99px;background:var(--panel);color:var(--muted);font-size:10px}.status.thinking{color:var(--accent)}.head-actions .secondary{min-height:36px;padding:7px 12px}
  .arena{min-height:0;flex:1;display:grid;grid-template-columns:minmax(420px,760px) minmax(330px,1fr);gap:20px;justify-content:center;align-items:start}.board-column{height:100%;min-height:0;display:flex;flex-direction:column;align-items:center;gap:6px}.board-tools{width:100%;display:flex;justify-content:flex-end;min-height:30px}.board-flip{width:34px;height:30px;padding:0;display:grid;place-items:center}.board-flip svg{width:17px;height:17px;fill:none;stroke:currentColor;stroke-linecap:round;stroke-linejoin:round;stroke-width:1.8}.board-frame{height:min(calc(100dvh - 260px),calc(100vw - 470px));max-height:760px;aspect-ratio:1}.outputs{height:min(calc(100dvh - 134px),760px);display:grid;grid-auto-rows:minmax(0,1fr);gap:10px;overflow:hidden}.human-match{padding:20px;border-radius:18px;border:1px solid var(--line);background:var(--panel)}.human-match p{color:var(--muted);font-size:12px;line-height:1.6}.loading-board{flex:1;display:grid;place-items:center;align-content:center;color:var(--muted)}.loading-board span{width:32px;aspect-ratio:1;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%}.error{margin-bottom:10px;padding:9px 12px;font-size:12px}
  @media(max-width:980px){:global(body.match-view){overflow:auto}.play-page{height:auto;min-height:calc(100dvh - 64px);overflow:visible}.arena{grid-template-columns:1fr}.board-frame{width:min(100%,calc(100dvh - 230px));height:auto}.outputs{height:auto;grid-template-columns:repeat(2,minmax(0,1fr));overflow:visible}}
  @media(max-width:680px){.play-page{padding:12px}.match-head{height:auto;align-items:flex-start}.match-title{align-items:flex-start;flex-direction:column;gap:4px}.head-actions{align-items:flex-end;flex-direction:column;gap:5px}.arena{display:block}.board-column{gap:5px}.board-frame{width:100%}.outputs{grid-template-columns:1fr;margin-top:12px}}
</style>
