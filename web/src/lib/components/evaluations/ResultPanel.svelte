<script lang="ts">
  import type { EvaluationJob } from '$lib/evaluations/types';
  import { evaluationsApi } from '$lib/api/evaluations';
  import PuzzleInsights from '$lib/components/evaluations/PuzzleInsights.svelte';
  let { job }: { job: EvaluationJob | undefined } = $props();
  const value = (key: string) => job?.result?.[key];
  const score = () => (value('score') as { wins?: number; draws?: number; losses?: number } | undefined) ?? {};
</script>
{#if job}<article class="result"><div class="result-head"><div><span class="eyebrow">Selected result</span><h2>{job.suite} evaluation</h2></div><span class="status {job.status}">{job.status}</span></div><p class="model">{job.candidate}{job.baseline ? ` vs ${job.baseline}` : ''}</p>
  {#if job.status === 'queued' || job.status === 'running'}<div class="progress"><span></span><p>The evaluator is running. This panel will update automatically.</p></div>
  {:else if job.status === 'failed' || job.status === 'interrupted'}<div class="error">{job.error ?? 'The evaluation did not complete.'}</div>
  {:else if job.result}
    {#if job.suite === 'puzzle'}
      <PuzzleInsights result={job.result} />
      <p class="caveat">Accuracy is measured on the selected local puzzle suite.</p>
    {:else}
      <div class="metrics"><div><b>{score().wins ?? 0} / {score().draws ?? 0} / {score().losses ?? 0}</b><small>W / D / L</small></div><div><b>{(Number(value('score_fraction') ?? 0) * 100).toFixed(1)}%</b><small>score fraction</small></div><div><b>{Number(value('smoothed_elo_delta') ?? 0).toFixed(1)}</b><small>local Elo estimate</small></div></div>
      <p class="caveat">Elo is a local estimate versus the named opponent and exact settings, not universal player Elo.</p>
    {/if}
  {/if}
  {#if job.artifacts.length}<div class="artifacts">{#each job.artifacts as artifact (artifact)}<button onclick={() => window.open(evaluationsApi.artifactUrl(job.id, artifact), '_blank')}>{artifact}</button>{/each}</div>{/if}
</article>{:else}<article class="result empty-result"><span class="eyebrow">Selected result</span><h2>Pick a job to inspect</h2><p>Launch an evaluation or choose one from history.</p></article>{/if}
<style>.result{padding:20px;border:1px solid var(--line);border-radius:16px;background:var(--panel)}h2{margin:7px 0 0;font-size:22px;text-transform:capitalize}.result-head{display:flex;justify-content:space-between;gap:14px}.status{padding:5px 8px;border-radius:7px;background:var(--panel-2);font-size:10px;text-transform:uppercase;color:var(--muted);height:max-content}.status.succeeded{color:var(--accent)}.status.failed,.status.interrupted{color:#ff9da7}.model,p{color:var(--muted);font-size:13px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:24px 0}.metrics div{padding:14px;border:1px solid var(--line);border-radius:11px}.metrics b{font-size:20px}.metrics small{display:block;color:var(--muted);margin-top:5px}.caveat{line-height:1.55}.error{margin:22px 0;padding:12px;border-radius:10px;background:rgba(255,96,112,.08);color:#ff9da7}.progress{margin:24px 0}.progress span{display:block;height:3px;background:var(--accent);animation:pulse 1.3s infinite}.progress p{line-height:1.5}.artifacts{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}.artifacts button{padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--accent);font-size:11px;cursor:pointer}.empty-result{min-height:210px;display:grid;align-content:center}.empty-result p{line-height:1.6}@keyframes pulse{50%{opacity:.35}}@media(max-width:700px){.metrics{grid-template-columns:1fr 1fr}.metrics div:last-child{grid-column:span 2}}</style>
