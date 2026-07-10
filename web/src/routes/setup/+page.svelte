<script lang="ts">
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { onMount } from 'svelte';
  import AgentCard from '$lib/components/AgentCard.svelte';
  import { loadAgents } from '$lib/config/agents';
  import { games } from '$lib/config/games';
  import { persistSelection, restoreSelection } from '$lib/state/selection';
  import type { AgentDescriptor, AppSelection } from '$lib/types';

  let agents = $state<AgentDescriptor[]>([]);
  let config = $state<AppSelection>(restoreSelection());
  let loading = $state(true);
  let error = $state('');
  const game = $derived(games.find((entry) => entry.id === config.gameId));
  const compatible = $derived(agents.filter((agent) => agent.games.includes(config.gameId)));

  onMount(() => { void initialize(); });

  async function initialize() {
    loading = true;
    error = '';
    try {
      agents = await loadAgents();
      const options = agents.filter((agent) => agent.games.includes(config.gameId));
      if (!options.length) throw new Error(`No ${game?.name ?? config.gameId} agents are configured.`);
      if (!options.some((agent) => agent.id === config.first.agentId)) config.first.agentId = options[0].id;
      if (!options.some((agent) => agent.id === config.second.agentId)) config.second.agentId = options.find((agent) => agent.kind !== 'human')?.id ?? options[0].id;
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      loading = false;
    }
  }

  function startMatch() {
    if (!config.first.agentId || !config.second.agentId) {
      error = 'Choose both players.';
      return;
    }
    persistSelection($state.snapshot(config));
    goto(resolve('/play'));
  }
</script>

<svelte:head><title>Choose players · Engine Zoo</title></svelte:head>

<main class="page setup-page screen-enter">
  <div class="crumb"><button onclick={() => goto(resolve('/'))}>← Change game</button><span>{game?.symbol} {game?.name}</span></div>
  <section class="intro">
    <span class="eyebrow">Step 2 of 2</span>
    <h1 class="hero-title">Choose players.</h1>
    <p class="lead">Available agents are loaded from <code>/config/agents.json</code>. Each side keeps its own engine settings.</p>
  </section>

  {#if error}<div class="error">{error}</div>{/if}
  {#if loading}
    <div class="loading-grid"><div></div><div></div></div>
  {:else}
    <section class="matchup">
      <AgentCard label={config.gameId === 'chess' ? 'White' : 'First'} agents={compatible} bind:selected={config.first.agentId} bind:behavior={config.first.behavior} />
      <div class="versus" aria-hidden="true">VS</div>
      <AgentCard label={config.gameId === 'chess' ? 'Black' : 'Second'} agents={compatible} bind:selected={config.second.agentId} bind:behavior={config.second.behavior} />
    </section>
  {/if}

  <div class="actions"><button class="secondary" onclick={() => goto(resolve('/'))}>Back</button><button class="primary" onclick={startMatch} disabled={loading || !!error}>Start match <span>→</span></button></div>
</main>

<style>
  .setup-page{padding-top:32px}.crumb{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:12px}.crumb button{border:0;background:transparent;color:var(--muted);cursor:pointer}.intro{max-width:760px;margin:38px 0 30px}.intro code{color:var(--text)}
  .matchup{display:grid;grid-template-columns:minmax(0,1fr) 52px minmax(0,1fr);gap:16px;align-items:center}.versus{width:46px;aspect-ratio:1;border-radius:50%;display:grid;place-items:center;border:1px solid var(--line);background:var(--panel);color:var(--muted);font-size:10px}
  .actions{display:flex;justify-content:flex-end;gap:12px;margin-top:24px}.primary span{margin-left:24px}.loading-grid{display:grid;grid-template-columns:1fr 1fr;gap:68px}.loading-grid div{height:330px;border-radius:22px;background:var(--panel)}
  @media(max-width:840px){.matchup{grid-template-columns:1fr}.versus{margin:auto}.loading-grid{grid-template-columns:1fr;gap:16px}.actions button{flex:1}}
</style>
