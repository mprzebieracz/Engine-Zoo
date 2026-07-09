<script lang="ts">
  import type { AgentDescriptor } from '$lib/types';
  let { label, agents, selected = $bindable() } = $props<{ label:string; agents:AgentDescriptor[]; selected:string }>();
  const current = $derived(agents.find((agent: AgentDescriptor)=>agent.id===selected));
</script>
<article class="agent-card">
  <span class="side-label">{label}</span>
  <select bind:value={selected}>{#each agents as agent}<option value={agent.id}>{agent.name}</option>{/each}</select>
  <div class="agent-icon">{current?.kind==='human'?'U':'α'}</div>
  <div class="agent-copy"><h2>{current?.name ?? 'Select agent'}</h2><p>{current?.description ?? ''}</p></div>
  {#if current?.badge}<span class="badge">{current.badge}</span>{/if}
  {#if current?.server}<code>{current.server}</code>{/if}
</article>
<style>
  .agent-card{min-height:290px;padding:26px;border:1px solid var(--line);border-radius:24px;background:linear-gradient(145deg,var(--panel),#10131a);position:relative}.side-label{color:var(--accent);text-transform:uppercase;letter-spacing:.12em;font-size:10px}.agent-card select{position:absolute;inset:20px 20px auto auto;max-width:52%;padding:9px 11px;border-radius:10px;border:1px solid var(--line);background:var(--panel-2);color:var(--text)}.agent-icon{width:72px;height:72px;border-radius:20px;display:grid;place-items:center;margin-top:46px;background:color-mix(in srgb,var(--accent) 12%,var(--panel-2));color:var(--accent);font-size:32px;font-weight:800}.agent-copy h2{font-size:25px;margin:20px 0 8px}.agent-copy p{color:var(--muted);line-height:1.55;min-height:48px}.badge{display:inline-block;padding:5px 8px;border-radius:99px;background:color-mix(in srgb,var(--accent) 10%,transparent);color:var(--accent);font-size:10px}.agent-card code{display:block;margin-top:13px;color:var(--muted);font-size:10px}@media(max-width:800px){.agent-card select{position:static;max-width:none;width:100%;margin-top:15px}.agent-icon{margin-top:25px}}
</style>
