<script lang="ts">
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';
  import { loadAgents } from '$lib/config/agents';
  import { games } from '$lib/config/games';
  import { persistSelection, restoreSelection } from '$lib/state/selection';
  import type { AgentDescriptor, AppSelection } from '$lib/types';
  import AgentCard from '$lib/components/AgentCard.svelte';

  let agents = $state<AgentDescriptor[]>([]);
  let config = $state<AppSelection>(restoreSelection());
  let error = $state('');
  const game = $derived(games.find((entry) => entry.id === config.gameId));
  const compatible = $derived(agents.filter((agent) => agent.games.includes(config.gameId)));
  const first = $derived(compatible.find((agent) => agent.id === config.firstAgentId));
  const second = $derived(compatible.find((agent) => agent.id === config.secondAgentId));

  onMount(async () => {
    config = restoreSelection();
    agents = await loadAgents();
    const options = agents.filter((agent) => agent.games.includes(config.gameId));
    if (!options.some((agent) => agent.id === config.firstAgentId)) config.firstAgentId = options.find((a) => a.kind === 'human')?.id ?? options[0]?.id ?? '';
    if (!options.some((agent) => agent.id === config.secondAgentId)) config.secondAgentId = options.find((a) => a.kind !== 'human')?.id ?? options[0]?.id ?? '';
  });

  function continueToGame() {
    error = '';
    if (!first || !second) { error = 'Select both players.'; return; }
    if ([first, second].filter((agent) => agent.kind === 'human').length !== 1) {
      error = 'The current Rust session API supports exactly one human and one engine.';
      return;
    }
    persistSelection(config);
    goto('/play');
  }
</script>
<svelte:head><title>Configure players · Engine Zoo</title></svelte:head>
<main class="page screen-enter">
  <div class="crumb"><button onclick={()=>goto('/')}>← Change game</button><span>{game?.symbol} {game?.name}</span></div>
  <section class="intro"><span class="eyebrow">Step 2 of 3</span><h1 class="hero-title">Configure the match.</h1><p class="lead">Choose who controls each side. Available agents are loaded from <code>static/config/agents.json</code>.</p></section>
  {#if error}<div class="error">{error}</div>{/if}
  <section class="matchup">
    <AgentCard label={config.gameId==='chess'?'White':'First'} agents={compatible} bind:selected={config.firstAgentId} />
    <div class="versus"><span>VS</span></div>
    <AgentCard label={config.gameId==='chess'?'Black':'Second'} agents={compatible} bind:selected={config.secondAgentId} />
  </section>
  <section class="settings">
    <div><span class="eyebrow">Search settings</span><h2>Engine behavior</h2></div>
    <label><span>MCTS simulations</span><input type="number" min="1" max="100000" bind:value={config.simulations}></label>
    <label class="toggle"><input type="checkbox" bind:checked={config.showAnalysis}><i></i><span>Show live agent output</span></label>
  </section>
  <div class="actions"><button class="secondary" onclick={()=>goto('/')}>Back</button><button class="primary" onclick={continueToGame}>Start match <span>→</span></button></div>
</main>

<style>
  .crumb{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:12px}.crumb button{border:0;background:transparent;color:var(--muted);cursor:pointer}.intro{max-width:760px;margin:54px 0 42px}.intro code{color:var(--text)}.matchup{display:grid;grid-template-columns:1fr 62px 1fr;gap:18px;align-items:center}.agent-card{min-height:290px;padding:26px;border:1px solid var(--line);border-radius:24px;background:linear-gradient(145deg,var(--panel),#10131a);position:relative}.side-label{color:var(--accent);text-transform:uppercase;letter-spacing:.12em;font-size:10px}.agent-card select{position:absolute;inset:20px 20px auto auto;max-width:52%;padding:9px 11px;border-radius:10px;border:1px solid var(--line);background:var(--panel-2);color:var(--text)}.agent-icon{width:72px;height:72px;border-radius:20px;display:grid;place-items:center;margin-top:46px;background:color-mix(in srgb,var(--accent) 12%,var(--panel-2));color:var(--accent);font-size:32px;font-weight:800}.agent-copy h2{font-size:25px;margin:20px 0 8px}.agent-copy p{color:var(--muted);line-height:1.55;min-height:48px}.badge{display:inline-block;padding:5px 8px;border-radius:99px;background:color-mix(in srgb,var(--accent) 10%,transparent);color:var(--accent);font-size:10px}.agent-card code{display:block;margin-top:13px;color:var(--muted);font-size:10px}.versus{display:grid;place-items:center}.versus span{width:54px;height:54px;border-radius:50%;display:grid;place-items:center;border:1px solid var(--line);background:var(--panel);color:var(--muted);font-size:11px}.settings{display:grid;grid-template-columns:1fr auto auto;gap:30px;align-items:center;margin-top:24px;padding:24px;border:1px solid var(--line);border-radius:20px;background:var(--panel)}.settings h2{margin:6px 0 0}.settings label>span{display:block;color:var(--muted);font-size:11px;margin-bottom:8px}.settings input[type=number]{width:120px;padding:10px;border-radius:10px;border:1px solid var(--line);background:var(--panel-2);color:var(--text)}.toggle{display:flex;align-items:center;gap:10px}.toggle input{display:none}.toggle i{width:38px;height:21px;border-radius:99px;background:#303641;position:relative}.toggle i:after{content:'';position:absolute;width:15px;height:15px;left:3px;top:3px;border-radius:50%;background:white;transition:.2s}.toggle input:checked+i{background:color-mix(in srgb,var(--accent) 65%,#303641)}.toggle input:checked+i:after{transform:translateX(17px)}.toggle span{margin:0!important}.actions{display:flex;justify-content:flex-end;gap:12px;margin-top:28px}.primary span{margin-left:28px}@media(max-width:800px){.matchup{grid-template-columns:1fr}.versus{height:30px}.settings{grid-template-columns:1fr}.agent-card select{position:static;max-width:none;width:100%;margin-top:15px}.agent-icon{margin-top:25px}.actions button{flex:1}}
</style>
