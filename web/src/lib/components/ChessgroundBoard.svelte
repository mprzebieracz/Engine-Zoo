<script lang="ts">
  import { onMount } from 'svelte';
  import { Chessground } from '@lichess-org/chessground';
  import type { Api } from '@lichess-org/chessground/api';
  import type { Config } from '@lichess-org/chessground/config';
  import type { Key } from '@lichess-org/chessground/types';
  import { Chess } from 'chess.js';
  import { SvelteMap } from 'svelte/reactivity';
  import type { LegalMove } from '$lib/types';
  import '@lichess-org/chessground/assets/chessground.base.css';
  import '@lichess-org/chessground/assets/chessground.brown.css';
  import '@lichess-org/chessground/assets/chessground.cburnett.css';

  let { moves, legalMoves, disabled = false, orientation = 'white', onmove } = $props<{
    moves: string[];
    legalMoves: LegalMove[];
    disabled?: boolean;
    orientation?: 'white' | 'black';
    onmove: (move: string) => void;
  }>();

  let element: HTMLDivElement;
  let ground: Api | undefined;
  let promotion = $state<{ from: string; to: string; moves: LegalMove[] } | null>(null);

  function chessState() {
    const chess = new Chess();
    for (const uci of moves) {
      if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci)) continue;
      try { chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' }); } catch { /* backend remains authoritative */ }
    }
    return chess;
  }

  function destinations(): Map<Key, Key[]> {
    const result = new SvelteMap<Key, Key[]>();
    for (const entry of legalMoves) {
      const from = entry.move.slice(0, 2) as Key;
      const to = entry.move.slice(2, 4) as Key;
      const list = result.get(from) ?? [];
      if (!list.includes(to)) list.push(to);
      result.set(from, list);
    }
    return result;
  }

  function playOrChoosePromotion(orig: Key, dest: Key) {
    const candidates = legalMoves.filter((entry: LegalMove) => entry.move.startsWith(`${orig}${dest}`));
    if (candidates.length === 0) return;
    if (candidates.length === 1) onmove(candidates[0].move);
    else promotion = { from: orig, to: dest, moves: candidates };
  }

  function choosePromotion(move: LegalMove) {
    promotion = null;
    onmove(move.move);
  }

  function config(): Config {
    const chess = chessState();
    const last = moves.at(-1);
    return {
      fen: chess.fen(),
      lastMove: last ? [last.slice(0, 2) as Key, last.slice(2, 4) as Key] : undefined,
      orientation,
      turnColor: chess.turn() === 'w' ? 'white' : 'black',
      coordinates: true,
      animation: { enabled: true, duration: 220 },
      highlight: { lastMove: true, check: true },
      movable: {
        free: false,
        color: disabled || promotion ? undefined : (chess.turn() === 'w' ? 'white' : 'black'),
        dests: disabled || promotion ? new Map() : destinations(),
        showDests: true,
        events: {
          after: (orig, dest) => {
            playOrChoosePromotion(orig, dest);
          }
        }
      },
      draggable: { enabled: !disabled && !promotion, showGhost: true },
      selectable: { enabled: !disabled && !promotion }
    };
  }

  onMount(() => {
    ground = Chessground(element, config());
    return () => ground?.destroy();
  });

  $effect(() => {
    moves; legalMoves; disabled; orientation; promotion;
    ground?.set(config());
  });
</script>

<div class="board-shell" class:disabled>
  <div bind:this={element} class="cg-wrap"></div>
  {#if promotion}
    <div class="promotion-backdrop" role="button" tabindex="0" aria-label="Cancel promotion" onclick={() => (promotion = null)} onkeydown={(event) => { if (event.key === 'Escape') promotion = null; }}>
      <div class="promotion-picker" role="dialog" tabindex="-1" aria-label="Choose promotion" onclick={(event) => event.stopPropagation()} onkeydown={(event) => event.stopPropagation()}>
        <strong>Promote pawn</strong>
        <div class="promotion-options">
          {#each promotion.moves as move}
            <button class="promotion-option" type="button" onclick={() => choosePromotion(move)}>
              {({ q: '♛ Queen', r: '♜ Rook', b: '♝ Bishop', n: '♞ Knight' } as Record<string, string>)[move.move[4]] ?? move.move[4]}
            </button>
          {/each}
        </div>
        <button class="promotion-cancel" type="button" onclick={() => (promotion = null)}>Cancel</button>
      </div>
    </div>
  {/if}
</div>

<style>
  .board-shell { position: relative; width: 100%; aspect-ratio: 1; border-radius: 18px; overflow: hidden; box-shadow: 0 18px 50px rgba(0,0,0,.3); border: 1px solid rgba(255,255,255,.09); }
  .cg-wrap { width: 100%; height: 100%; }
  .disabled { opacity: .88; }
  .promotion-backdrop { position: absolute; inset: 0; z-index: 20; display: grid; place-items: center; background: rgba(15,20,14,.58); }
  .promotion-picker { display: grid; gap: .75rem; min-width: 13rem; padding: 1rem; border: 1px solid rgba(255,255,255,.2); border-radius: .9rem; background: #fffdf8; color: #1f2328; box-shadow: 0 18px 45px rgba(0,0,0,.28); text-align: center; }
  .promotion-options { display: grid; gap: .45rem; }
  .promotion-option, .promotion-cancel { border: 1px solid #ddd3c0; border-radius: .55rem; padding: .5rem .7rem; background: #f5f1e8; cursor: pointer; font: inherit; }
  .promotion-option:hover, .promotion-cancel:hover { border-color: #2f654f; background: #fff; }
  .promotion-cancel { font-size: .8rem; color: #6b6256; }
  :global(.cg-wrap cg-board) { border-radius: 16px; overflow: hidden; }
</style>
