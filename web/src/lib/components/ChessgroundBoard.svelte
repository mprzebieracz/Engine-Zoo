<script lang="ts">
  import { onMount } from 'svelte';
  import { Chessground } from '@lichess-org/chessground';
  import type { Api } from '@lichess-org/chessground/api';
  import type { Config } from '@lichess-org/chessground/config';
  import type { Key } from '@lichess-org/chessground/types';
  import { Chess } from 'chess.js';
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

  function state() {
    const chess = new Chess();
    for (const uci of moves) {
      if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci)) continue;
      try { chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' }); } catch { /* backend remains authoritative */ }
    }
    return chess;
  }

  function destinations(): Map<Key, Key[]> {
    const result = new Map<Key, Key[]>();
    for (const entry of legalMoves) {
      const from = entry.move.slice(0, 2) as Key;
      const to = entry.move.slice(2, 4) as Key;
      const list = result.get(from) ?? [];
      if (!list.includes(to)) list.push(to);
      result.set(from, list);
    }
    return result;
  }

  function config(): Config {
    const chess = state();
    return {
      fen: chess.fen(),
      orientation,
      turnColor: chess.turn() === 'w' ? 'white' : 'black',
      coordinates: true,
      animation: { enabled: true, duration: 220 },
      highlight: { lastMove: true, check: true },
      movable: {
        free: false,
        color: disabled ? undefined : (chess.turn() === 'w' ? 'white' : 'black'),
        dests: disabled ? new Map() : destinations(),
        showDests: true,
        events: {
          after: (orig, dest) => {
            const candidate = legalMoves.find((entry: LegalMove) => entry.move.startsWith(`${orig}${dest}`));
            if (candidate) onmove(candidate.move);
          }
        }
      },
      draggable: { enabled: !disabled, showGhost: true },
      selectable: { enabled: !disabled }
    };
  }

  onMount(() => {
    ground = Chessground(element, config());
    return () => ground?.destroy();
  });

  $effect(() => {
    moves; legalMoves; disabled; orientation;
    ground?.set(config());
  });
</script>

<div class="board-shell" class:disabled>
  <div bind:this={element} class="cg-wrap"></div>
</div>

<style>
  .board-shell { width: min(74vh, 100%); aspect-ratio: 1; border-radius: 22px; overflow: hidden; box-shadow: 0 32px 90px rgba(0,0,0,.38); border: 1px solid rgba(255,255,255,.09); }
  .cg-wrap { width: 100%; height: 100%; }
  .disabled { opacity: .88; }
  :global(.cg-wrap cg-board) { border-radius: 20px; overflow: hidden; }
</style>
