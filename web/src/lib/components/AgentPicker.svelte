<script lang="ts">
  import type { AgentDescriptor } from '$lib/types';

  let { label, agents, selected, onchange } = $props<{
    label: string;
    agents: AgentDescriptor[];
    selected: string;
    onchange: (id: string) => void;
  }>();

  const current = $derived(agents.find((agent: AgentDescriptor) => agent.id === selected));
</script>

<label class="picker">
  <span class="label">{label}</span>
  <select value={selected} onchange={(event) => onchange(event.currentTarget.value)}>
    {#each agents as agent}
      <option value={agent.id}>{agent.name}</option>
    {/each}
  </select>
  <small>{current?.description ?? 'Configured agent'}</small>
</label>

<style>
  .picker { display: block; padding: 16px; border: 1px solid var(--line); border-radius: 16px; background: var(--panel-2); }
  .label { display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .09em; margin-bottom: 9px; }
  select { width: 100%; padding: 11px 12px; border-radius: 10px; border: 1px solid var(--line); background: var(--panel); color: var(--text); outline: none; }
  select:focus { border-color: var(--accent); }
  small { display: block; color: var(--muted); margin-top: 9px; line-height: 1.4; min-height: 34px; }
</style>
