<script lang="ts">
  import type { PreflightState, Suite } from '$lib/evaluations/types';
  let { value, preflight, onChange }: { value: Suite; preflight: Record<Suite, PreflightState>; onChange: (value: Suite) => void } = $props();
  const suites: { id: Suite; title: string; copy: string }[] = [
    { id: 'puzzle', title: 'Puzzle', copy: 'Deterministic best-move accuracy on the smoke suite.' },
    { id: 'stockfish', title: 'Stockfish', copy: 'Fixed-node games against the local Stockfish reference.' },
    { id: 'arena', title: 'Arena', copy: 'Paired openings against another saved checkpoint.' }
  ];
</script>
<div class="suite-grid">{#each suites as suite (suite.id)}
  {@const state = preflight[suite.id]}
  <button class:selected={value === suite.id} class:unavailable={!state?.available} class="suite" onclick={() => onChange(suite.id)} disabled={!state?.available}>
    <span class="radio">{value === suite.id ? '●' : '○'}</span><strong>{suite.title}</strong><span class="copy">{suite.copy}</span><small>{state?.available ? 'Ready' : state?.message}</small>
  </button>
{/each}</div>
<style>.suite-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.suite{min-height:145px;text-align:left;padding:16px;border:1px solid var(--line);border-radius:15px;background:var(--panel);display:flex;flex-direction:column;gap:9px;cursor:pointer}.suite.selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}.suite:disabled{cursor:not-allowed;opacity:.55}.radio{color:var(--accent);font-size:15px}.copy,small{color:var(--muted);font-size:12px;line-height:1.45}small{margin-top:auto;color:var(--accent)}.unavailable small{color:#ff9da7}@media(max-width:700px){.suite-grid{grid-template-columns:1fr}}</style>
