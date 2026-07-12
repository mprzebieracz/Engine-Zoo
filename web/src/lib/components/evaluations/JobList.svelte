<script lang="ts">
  import type { EvaluationJob } from '$lib/evaluations/types';
  let { jobs, selected, onSelect }: { jobs: EvaluationJob[]; selected: string; onSelect: (id: string) => void } = $props();
  const date = (value: number) => new Date(value * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
</script>
<div class="list">{#if !jobs.length}<p class="empty">No evaluations yet. Completed results stay beside the training run.</p>{/if}{#each jobs as job (job.id)}
  <button class:selected={selected === job.id} onclick={() => onSelect(job.id)}><span><strong>{job.suite}</strong><small>{job.candidate} · {date(job.created_at)}</small></span><em class={job.status}>{job.status}</em></button>
{/each}</div>
<style>.list{display:grid;gap:7px}.list button{display:flex;justify-content:space-between;align-items:center;gap:12px;text-align:left;padding:13px;border:1px solid var(--line);border-radius:12px;background:var(--panel);cursor:pointer}.list button.selected{border-color:var(--accent)}strong{display:block;text-transform:capitalize}small{display:block;color:var(--muted);font-size:11px;margin-top:4px}em{font-size:10px;font-style:normal;color:var(--muted)}em.succeeded{color:var(--accent)}em.failed,em.interrupted{color:#ff9da7}.empty{color:var(--muted);font-size:13px;line-height:1.6}</style>
