<script lang="ts">
  import type { LegalMove } from '$lib/types';

  let { moves, legalMoves, disabled = false, onmove } = $props<{
    moves: string[];
    legalMoves: LegalMove[];
    disabled?: boolean;
    onmove: (move: string) => void;
  }>();

  const initial: Record<string, string> = {
    a8:'♜',b8:'♞',c8:'♝',d8:'♛',e8:'♚',f8:'♝',g8:'♞',h8:'♜',
    a7:'♟',b7:'♟',c7:'♟',d7:'♟',e7:'♟',f7:'♟',g7:'♟',h7:'♟',
    a2:'♙',b2:'♙',c2:'♙',d2:'♙',e2:'♙',f2:'♙',g2:'♙',h2:'♙',
    a1:'♖',b1:'♘',c1:'♗',d1:'♕',e1:'♔',f1:'♗',g1:'♘',h1:'♖'
  };
  const files = ['a','b','c','d','e','f','g','h'];
  const squares = Array.from({ length: 64 }, (_, i) => `${files[i % 8]}${8 - Math.floor(i / 8)}`);
  let selected = $state<string | null>(null);

  function boardFromMoves() {
    const board = { ...initial };
    for (const move of moves) {
      if (!/^[a-h][1-8][a-h][1-8]/.test(move)) continue;
      const from = move.slice(0, 2);
      const to = move.slice(2, 4);
      const piece = board[from];
      if (!piece) continue;
      delete board[from];
      board[to] = piece;
      if (move.length >= 5) {
        const white = piece === piece.toUpperCase();
        const promoted: Record<string, [string,string]> = { q:['♛','♕'], r:['♜','♖'], b:['♝','♗'], n:['♞','♘'] };
        const pair = promoted[move[4].toLowerCase()];
        if (pair) board[to] = white ? pair[1] : pair[0];
      }
    }
    return board;
  }

  const board = $derived(boardFromMoves());
  const legalFromSelected = $derived(new Set(legalMoves.filter((m: LegalMove) => m.move.startsWith(selected ?? '---')).map((m: LegalMove) => m.move.slice(2,4))));

  function clickSquare(square: string) {
    if (disabled) return;
    if (selected) {
      const candidates = legalMoves.filter((entry: LegalMove) => entry.move.startsWith(selected + square));
      if (candidates.length) {
        onmove(candidates[0].move);
        selected = null;
        return;
      }
    }
    const hasLegal = legalMoves.some((entry: LegalMove) => entry.move.startsWith(square));
    selected = hasLegal ? square : null;
  }
</script>

<div class:disabled class="board" aria-label="Chess board">
  {#each squares as square, i}
    <button
      class:dark={(Math.floor(i / 8) + i % 8) % 2 === 1}
      class:selected={selected === square}
      class:target={legalFromSelected.has(square)}
      class="square"
      onclick={() => clickSquare(square)}
      aria-label={square}
    >
      <span class="piece">{board[square] ?? ''}</span>
      {#if i % 8 === 0}<small class="rank">{square[1]}</small>{/if}
      {#if Math.floor(i / 8) === 7}<small class="file">{square[0]}</small>{/if}
    </button>
  {/each}
</div>

<style>
  .board { width: min(72vh, 100%); aspect-ratio: 1; display: grid; grid-template-columns: repeat(8, 1fr); border-radius: 20px; overflow: hidden; box-shadow: 0 28px 70px rgba(0,0,0,.34); border: 1px solid rgba(255,255,255,.08); }
  .square { position: relative; border: 0; padding: 0; background: #d7dfc0; cursor: pointer; color: #172016; }
  .square.dark { background: #71845e; }
  .square.selected { box-shadow: inset 0 0 0 5px rgba(183,243,107,.8); }
  .square.target::after { content: ''; position: absolute; inset: 37%; border-radius: 50%; background: rgba(25,33,22,.35); }
  .piece { position: relative; z-index: 2; font-size: clamp(28px, 6.1vw, 68px); line-height: 1; filter: drop-shadow(0 3px 1px rgba(0,0,0,.24)); transition: transform .16s ease; }
  .square:hover .piece { transform: translateY(-2px); }
  .disabled { pointer-events: none; opacity: .8; }
  small { position: absolute; font: 700 10px/1 system-ui; opacity: .65; }
  .rank { top: 4px; left: 5px; } .file { right: 5px; bottom: 4px; }
</style>
