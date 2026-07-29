<script lang="ts">
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { games } from '$lib/config/games';
  import { persistSelection, restoreSelection } from '$lib/state/selection';
  import { onMount } from 'svelte';
  let selected = $state('chess');
  let index = $derived(Math.max(0, games.findIndex((game) => game.id === selected)));

  onMount(() => {
    selected = restoreSelection().gameId;
  });

  function cycle(delta: number) {
    let next = index;

    do {
      next = (next + delta + games.length) % games.length;
    } while (!games[next].available);

    selected = games[next].id;
  }

  function next() {
    const selection = restoreSelection();
    persistSelection({ ...selection, gameId: selected });

    goto(resolve('/setup'));
  }
</script>
<svelte:head><title>Choose game · Engine Zoo</title></svelte:head>
<main class="page screen-enter">
  <section class="intro"><span class="eyebrow">Step 1 of 2</span><h1 class="hero-title">Choose the arena.</h1><p class="lead">Select a board game, then choose the players on the next screen.</p></section>
  <section class="stage">
    <button class="arrow" onclick={()=>cycle(-1)} aria-label="Previous game">←</button>
    <div class="viewport">
      {#each games as game, i (game.id)}
        <button class="game" class:active={game.id===selected} class:side={Math.abs(i-index)===1} class:hidden={Math.abs(i-index)>1} class:disabled={!game.available} style={`--game-accent:${game.accent};--offset:${i-index}`} onclick={()=>game.available&&(selected=game.id)} disabled={!game.available}>
          <span class="symbol">{game.symbol}</span><h2>{game.name}</h2>
        </button>
      {/each}
    </div>
    <button class="arrow" onclick={()=>cycle(1)} aria-label="Next game">→</button>
  </section>
  <div class="footer"><div class="dots">{#each games.filter(g=>g.available) as game (game.id)}<button class:active={game.id===selected} onclick={()=>selected=game.id} aria-label={game.name}></button>{/each}</div><button class="primary" onclick={next}>Next step <span>→</span></button></div>
</main>
<style>
  .intro{text-align:center;max-width:780px;margin:18px auto 56px}.intro .lead{margin:auto}.stage{display:grid;grid-template-columns:48px minmax(0,1fr) 48px;align-items:center;gap:16px}.viewport{height:330px;position:relative;overflow:hidden}.game{position:absolute;left:50%;top:50%;width:min(520px,72vw);height:270px;transform:translate(calc(-50% + var(--offset)*72%), -50%) scale(.82);opacity:.42;border:1px solid var(--line);border-radius:30px;background:var(--panel);padding:38px;text-align:left;transition:transform .18s ease-out,opacity .18s ease-out,border-color .18s ease-out;overflow:hidden;cursor:pointer}.game.active{transform:translate(-50%,-50%) scale(1);opacity:1;border-color:color-mix(in srgb,var(--game-accent) 60%,var(--line));z-index:3}.game.hidden{opacity:0;pointer-events:none}.game.disabled{filter:grayscale(.7)}.symbol{width:76px;height:76px;border-radius:22px;display:grid;place-items:center;background:color-mix(in srgb,var(--game-accent) 13%,var(--panel-2));color:var(--game-accent);font-size:42px}.game h2{font-size:35px;margin:30px 0 0}.arrow{width:46px;height:46px;border-radius:50%;border:1px solid var(--line);background:var(--panel);cursor:pointer;font-size:18px}.footer{display:flex;align-items:center;justify-content:space-between;margin-top:32px;max-width:800px;margin-left:auto;margin-right:auto}.primary span{margin-left:25px}.dots{display:flex;gap:7px}.dots button{width:7px;height:7px;padding:0;border:0;border-radius:50%;background:#3a404b}.dots button.active{width:24px;border-radius:99px;background:var(--accent)}@media(max-width:700px){.stage{grid-template-columns:44px 1fr 44px}.viewport{height:310px}.game{width:78vw;height:250px;padding:28px}.game.side{opacity:0}.arrow{width:44px;height:44px}.footer{justify-content:center;gap:24px;flex-direction:column}.primary{width:100%}}
</style>
