<script lang="ts">
  import type { AgentBehavior, AgentDescriptor } from '$lib/types';

  let { label, agents, selected = $bindable(), behavior = $bindable() } = $props<{
    label: string;
    agents: AgentDescriptor[];
    selected: string;
    behavior: AgentBehavior;
  }>();

  const current = $derived(agents.find((agent: AgentDescriptor) => agent.id === selected));

  function selectAgent(event: Event) {
    selected = (event.currentTarget as HTMLSelectElement).value;
    const agent = agents.find((entry: AgentDescriptor) => entry.id === selected);
    if (agent?.kind === 'alphazero') {
      behavior.simulations = agent.defaults?.simulations ?? 800;
      behavior.waitForCount = agent.defaults?.waitForCount ?? 1;
    }
  }
</script>

<article class="agent-card">
  <header>
    <span class="side-label">{label}</span>
    <select value={selected} onchange={selectAgent} aria-label={`${label} agent`}>
      {#each agents as agent (agent.id)}<option value={agent.id}>{agent.name}</option>{/each}
    </select>
  </header>

  <div class="identity">
    <div class="agent-icon">{current?.kind === 'human' ? 'U' : 'α'}</div>
    <div><h2>{current?.name ?? 'Select agent'}</h2><p>{current?.description ?? 'Choose an agent for this side.'}</p></div>
  </div>

  {#if current?.kind === 'alphazero'}
    <section class="behavior" aria-label={`${label} AlphaZero behavior`}>
      <div class="section-heading"><strong>Engine behavior</strong><small>{current.server}</small></div>
      <div class="setting-grid">
        <label><span>MCTS simulations</span><input type="number" min="1" max="100000" bind:value={behavior.simulations} /></label>
        <label><span>Move policy</span><select bind:value={behavior.movePolicy}><option value="strict">Strict best move</option><option value="temperature">Temperature sampling</option></select></label>
        {#if behavior.movePolicy === 'temperature'}
          <label><span>Temperature</span><input type="number" min="0.05" max="5" step="0.05" bind:value={behavior.temperature} /></label>
        {/if}
      </div>
    </section>
  {:else}
    <div class="human-note">The match waits for this player to make a move on the board.</div>
  {/if}
</article>

<style>
  .agent-card{min-height:330px;padding:24px;border:1px solid var(--line);border-radius:22px;background:var(--panel);display:flex;flex-direction:column;gap:22px}
  header,.section-heading{display:flex;align-items:center;justify-content:space-between;gap:16px}.side-label{color:var(--accent);font-size:11px;font-weight:750;text-transform:uppercase}
  select,input{min-height:42px;padding:9px 11px;border-radius:10px;border:1px solid var(--line);background:var(--panel-2);color:var(--text);outline:none}select:focus,input:focus{border-color:var(--accent)}
  header select{max-width:62%}.identity{display:grid;grid-template-columns:62px 1fr;gap:16px;align-items:center}.agent-icon{width:62px;aspect-ratio:1;border-radius:17px;display:grid;place-items:center;background:color-mix(in srgb,var(--accent) 12%,var(--panel-2));color:var(--accent);font-size:28px;font-weight:800}
  h2{font-size:22px;margin:0 0 6px}.identity p{color:var(--muted);font-size:13px;line-height:1.5;margin:0}.behavior{margin-top:auto;padding-top:18px;border-top:1px solid var(--line)}.section-heading strong{font-size:13px}.section-heading small{max-width:58%;color:var(--muted);font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .setting-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:14px}.setting-grid label>span{display:block;color:var(--muted);font-size:10px;margin-bottom:7px}.setting-grid input,.setting-grid select{width:100%}.human-note{margin-top:auto;padding:14px;border-radius:12px;background:var(--panel-2);color:var(--muted);font-size:12px;line-height:1.5}
  @media(max-width:620px){header{align-items:stretch;flex-direction:column}header select{max-width:none;width:100%}.setting-grid{grid-template-columns:1fr}}
</style>
