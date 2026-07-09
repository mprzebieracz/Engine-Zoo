<script lang="ts">
  import { analyze, createChessSession, playMove, type Analysis, type SessionView } from '$lib/api';

  let session: SessionView | null = null;
  let model = 'best';
  let simulations = 800;
  let move = '';
  let status = 'Ready';
  let thinking = false;
  let analysis: Analysis | null = null;
  let error = '';

  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

  function boardCells(board: string): string[] {
    const lines = board.split('\n').slice(0, 8);
    return lines.flatMap((line) => line.trim().split(/\s+/).slice(1, 9));
  }

  function squareName(index: number): string {
    const rank = 8 - Math.floor(index / 8);
    const file = files[index % 8];
    return `${file}${rank}`;
  }

  async function newGame(engineFirst = false) {
    thinking = true;
    error = '';
    status = engineFirst ? 'Engine opening' : 'White to move';
    try {
      session = await createChessSession({ engineFirst, model, simulations });
      analysis = null;
      status = session.terminal ? 'Game over' : session.human_turn ? 'Your move' : 'Engine to move';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      status = 'Error';
    } finally {
      thinking = false;
    }
  }

  async function submitMove() {
    if (!session || !move.trim()) return;
    thinking = true;
    error = '';
    status = 'Engine thinking';
    try {
      session = await playMove(session.id, move.trim());
      move = '';
      status = session.terminal ? 'Game over' : session.human_turn ? 'Your move' : 'Engine to move';
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      status = 'Move rejected';
    } finally {
      thinking = false;
    }
  }

  async function analyzeCurrent(mode: 'net' | 'mcts') {
    if (!session) return;
    thinking = true;
    error = '';
    try {
      analysis = await analyze({
        position: { game: 'chess', position: { moves: session.moves } },
        model,
        mode,
        simulations,
        wait_for_count: 1
      });
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      thinking = false;
    }
  }

  $: cells = session ? boardCells(session.board) : Array(64).fill('');
</script>

<svelte:head>
  <title>engine-zoo chess</title>
  <meta name="description" content="Play and analyze chess against engine-zoo agents" />
</svelte:head>

<main class="shell">
  <nav class="topbar">
    <div>
      <a class="brand" href="/">engine-zoo</a>
      <span>Chess</span>
      <a href="/connect4">Connect4</a>
    </div>
    <div class="controls">
      <label>
        Model
        <input bind:value={model} />
      </label>
      <label>
        Sims
        <input type="number" min="1" bind:value={simulations} />
      </label>
      <button on:click={() => newGame(false)}>New</button>
      <button class="secondary" on:click={() => newGame(true)}>Engine First</button>
    </div>
  </nav>

  <section class="workspace">
    <section class="board-panel">
      <div class="status-row">
        <div>
          <strong>{status}</strong>
          <span>{session ? `session ${session.id}` : 'no session'}</span>
        </div>
        {#if thinking}<i>thinking</i>{/if}
      </div>

      <div class="board" aria-label="Chess board">
        {#each cells as piece, index}
          <button
            class:dark={(Math.floor(index / 8) + index) % 2 === 1}
            class="square"
            title={squareName(index)}
            type="button"
          >
            <span>{piece === '.' ? '' : piece}</span>
            <small>{index % 8 === 0 ? 8 - Math.floor(index / 8) : ''}</small>
            <em>{index >= 56 ? files[index % 8] : ''}</em>
          </button>
        {/each}
      </div>

      <form class="move-form" on:submit|preventDefault={submitMove}>
        <input bind:value={move} placeholder="e2e4, g1f3, e7e8q" autocomplete="off" />
        <button disabled={!session || thinking}>Play</button>
      </form>
      {#if error}<p class="error">{error}</p>{/if}
    </section>

    <aside class="side">
      <section class="panel">
        <div class="panel-head">
          <h2>Position</h2>
          <div>
            <button class="secondary" disabled={!session || thinking} on:click={() => analyzeCurrent('net')}>Net</button>
            <button disabled={!session || thinking} on:click={() => analyzeCurrent('mcts')}>MCTS</button>
          </div>
        </div>
        <div class="value">
          <span>Value</span>
          <strong>{analysis ? analysis.value.toFixed(3) : '-'}</strong>
        </div>
        <div class="value">
          <span>Best</span>
          <strong>{analysis?.best_move ?? '-'}</strong>
        </div>
        <div class="policy">
          {#each (analysis?.policy ?? []).slice(0, 10) as item}
            <div>
              <span>{item.mv}</span>
              <i style={`width: ${Math.max(2, item.p * 100)}%`}></i>
              <strong>{(item.p * 100).toFixed(1)}%</strong>
            </div>
          {/each}
        </div>
      </section>

      <section class="panel">
        <h2>Legal Moves</h2>
        <div class="moves">
          {#each session?.legal_moves ?? [] as legal}
            <button class="chip" type="button" on:click={() => (move = legal.move)}>{legal.move}</button>
          {/each}
        </div>
      </section>

      <section class="panel">
        <h2>Game</h2>
        <pre>{session?.pgn || '-'}</pre>
      </section>
    </aside>
  </section>
</main>

<style>
  :global(body) {
    margin: 0;
    background: #f4f0e7;
    color: #20231f;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  .shell {
    min-height: 100vh;
    padding: 22px;
  }

  .topbar {
    display: flex;
    gap: 18px;
    justify-content: space-between;
    align-items: end;
    border-bottom: 1px solid #d7cdbc;
    padding-bottom: 18px;
  }

  .topbar > div {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .brand {
    font-size: 24px;
    font-weight: 800;
    color: #20231f;
    text-decoration: none;
  }

  a {
    color: #37624f;
    font-weight: 700;
  }

  .controls {
    flex-wrap: wrap;
  }

  label {
    display: grid;
    gap: 4px;
    color: #696153;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
  }

  input {
    border: 1px solid #cfc3b0;
    border-radius: 7px;
    background: #fffdfa;
    color: #20231f;
    font: inherit;
    padding: 10px 12px;
  }

  button {
    border: 0;
    border-radius: 7px;
    background: #37624f;
    color: white;
    cursor: pointer;
    font: inherit;
    font-weight: 800;
    padding: 10px 14px;
  }

  button.secondary,
  .chip {
    border: 1px solid #cfc3b0;
    background: #fffdfa;
    color: #37624f;
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }

  .workspace {
    display: grid;
    grid-template-columns: minmax(360px, 680px) minmax(320px, 1fr);
    gap: 22px;
    margin-top: 22px;
  }

  .board-panel,
  .panel {
    border: 1px solid #d7cdbc;
    border-radius: 8px;
    background: #fffdfa;
    padding: 18px;
  }

  .status-row,
  .panel-head,
  .value {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
  }

  .status-row span,
  .value span {
    color: #756d61;
    font-size: 13px;
  }

  .board {
    aspect-ratio: 1;
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    margin: 18px auto;
    max-width: 640px;
    overflow: hidden;
    border-radius: 8px;
    box-shadow: 0 18px 60px rgb(47 42 33 / 18%);
  }

  .square {
    position: relative;
    aspect-ratio: 1;
    border-radius: 0;
    background: #e2d5bb;
    color: #191a17;
    display: grid;
    place-items: center;
    padding: 0;
  }

  .square.dark {
    background: #78906b;
  }

  .square span {
    font-family: Georgia, serif;
    font-size: clamp(26px, 6vw, 52px);
    font-weight: 700;
    line-height: 1;
  }

  .square small,
  .square em {
    position: absolute;
    color: rgb(0 0 0 / 48%);
    font-size: 11px;
    font-style: normal;
    font-weight: 800;
  }

  .square small {
    left: 5px;
    top: 4px;
  }

  .square em {
    bottom: 3px;
    right: 5px;
  }

  .move-form {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 10px;
  }

  .side {
    display: grid;
    align-content: start;
    gap: 16px;
  }

  h2 {
    font-size: 17px;
    margin: 0 0 12px;
  }

  .policy {
    display: grid;
    gap: 9px;
    margin-top: 14px;
  }

  .policy div {
    display: grid;
    grid-template-columns: 58px 1fr 58px;
    gap: 10px;
    align-items: center;
    font-size: 13px;
  }

  .policy i {
    display: block;
    height: 9px;
    border-radius: 999px;
    background: linear-gradient(90deg, #37624f, #b98d3d);
  }

  .moves {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    max-height: 210px;
    overflow: auto;
  }

  pre {
    white-space: pre-wrap;
    word-break: break-word;
    color: #3b372f;
    font-size: 13px;
    line-height: 1.6;
  }

  .error {
    color: #a64038;
    font-weight: 700;
  }

  @media (max-width: 980px) {
    .topbar,
    .workspace {
      grid-template-columns: 1fr;
      display: grid;
    }

    .topbar > div {
      align-items: stretch;
      flex-wrap: wrap;
    }
  }
</style>
