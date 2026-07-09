<script lang="ts">
  let { moves, disabled = false, onmove } = $props<{ moves: string[]; disabled?: boolean; onmove: (move: string) => void }>();
  const cells = Array.from({ length: 42 }, (_, i) => i);
  function occupant(index: number) {
    const row = Math.floor(index / 7);
    const col = index % 7;
    for (let i = moves.length - 1; i >= 0; i--) {
      if (Number(moves[i]) !== col) continue;
      const prior = moves.slice(0, i).filter((m: string) => Number(m) === col).length;
      const placedRow = 5 - prior;
      if (placedRow === row) return i % 2 === 0 ? 'first' : 'second';
    }
    return '';
  }
</script>

<div class:disabled class="connect" aria-label="Connect Four board">
  {#each cells as cell}
    {@const col = cell % 7}
    {@const owner = occupant(cell)}
    <button aria-label={`Column ${col + 1}`} onclick={() => onmove(String(col))}>
      <span class:filled={!!owner} class:first={owner === 'first'} class:second={owner === 'second'}></span>
    </button>
  {/each}
</div>

<style>
  .connect { width: min(760px, 100%); aspect-ratio: 7 / 6; display: grid; grid-template-columns: repeat(7, 1fr); padding: 18px; gap: 10px; border-radius: 24px; background: #2d63c7; box-shadow: 0 28px 70px rgba(0,0,0,.34); }
  button { border: 0; background: transparent; padding: 0; cursor: pointer; }
  span { display: block; width: 100%; aspect-ratio: 1; border-radius: 50%; background: #0d1119; box-shadow: inset 0 7px 12px rgba(0,0,0,.45); transition: transform .2s ease, background .2s ease; }
  button:hover span { transform: scale(.92); }
  span.first { background: #ffcc4d; } span.second { background: #ef5b65; }
  .disabled { pointer-events: none; opacity: .8; }
</style>
