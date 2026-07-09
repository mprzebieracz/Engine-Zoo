<script lang="ts">
  import type { GameDescriptor } from '$lib/types';

  let { games, selected, onselect } = $props<{
    games: GameDescriptor[];
    selected: string;
    onselect: (id: string) => void;
  }>();

  const available = $derived(games.filter((game: GameDescriptor) => game.available));
  const selectedIndex = $derived(Math.max(0, available.findIndex((game: GameDescriptor) => game.id === selected)));

  function cycle(direction: number) {
    const index = (selectedIndex + direction + available.length) % available.length;
    onselect(available[index].id);
  }
</script>

<section class="carousel" aria-label="Game selection">
  <button class="arrow" onclick={() => cycle(-1)} aria-label="Previous game">←</button>
  <div class="cards">
    {#each games as game (game.id)}
      <button
        class:active={game.id === selected}
        class:disabled={!game.available}
        class="game-card"
        style={`--accent: ${game.accent}`}
        onclick={() => game.available && onselect(game.id)}
        disabled={!game.available}
      >
        <div class="symbol">{game.symbol}</div>
        <div>
          <strong>{game.name}</strong>
          <span>{game.description}</span>
        </div>
        {#if !game.available}<small>Later</small>{/if}
      </button>
    {/each}
  </div>
  <button class="arrow" onclick={() => cycle(1)} aria-label="Next game">→</button>
</section>

<style>
  .carousel { display: grid; grid-template-columns: 44px 1fr 44px; gap: 14px; align-items: center; }
  .cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
  .game-card { min-height: 132px; text-align: left; display: flex; align-items: center; gap: 16px; padding: 20px; border-radius: 20px; border: 1px solid var(--line); background: var(--panel); color: var(--text); cursor: pointer; transition: transform .22s ease, border-color .22s ease, background .22s ease, opacity .22s ease; position: relative; overflow: hidden; }
  .game-card::after { content: ''; position: absolute; inset: auto -40px -70px auto; width: 140px; height: 140px; border-radius: 50%; background: color-mix(in srgb, var(--accent) 23%, transparent); filter: blur(12px); transition: transform .3s ease; }
  .game-card:hover:not(:disabled) { transform: translateY(-3px); border-color: color-mix(in srgb, var(--accent) 60%, var(--line)); }
  .game-card.active { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 7%, var(--panel)); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 25%, transparent); }
  .game-card.active::after { transform: scale(1.3); }
  .game-card.disabled { opacity: .42; cursor: not-allowed; }
  .symbol { width: 60px; height: 60px; display: grid; place-items: center; border-radius: 17px; background: color-mix(in srgb, var(--accent) 13%, var(--panel-2)); color: var(--accent); font-size: 34px; flex: 0 0 auto; }
  strong, span { display: block; }
  strong { font-size: 16px; margin-bottom: 7px; }
  span { color: var(--muted); font-size: 13px; line-height: 1.45; }
  small { position: absolute; top: 12px; right: 12px; color: var(--muted); }
  .arrow { height: 44px; width: 44px; border-radius: 50%; border: 1px solid var(--line); background: var(--panel); color: var(--text); cursor: pointer; transition: transform .2s ease, background .2s ease; }
  .arrow:hover { transform: scale(1.05); background: var(--panel-2); }
  @media (max-width: 900px) { .cards { grid-template-columns: 1fr; } .game-card:not(.active) { display: none; } }
  @media (max-width: 560px) { .carousel { grid-template-columns: 34px 1fr 34px; gap: 8px; } .arrow { width: 34px; height: 34px; } }
</style>
