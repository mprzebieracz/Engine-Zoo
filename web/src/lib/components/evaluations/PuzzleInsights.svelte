<script lang="ts">
  type Stats = { correct?: number; total?: number; accuracy?: number };
  type Breakdown = { name: string; correct: number; total: number; accuracy: number };

  let { result }: { result: Record<string, unknown> } = $props();

  const number = (value: unknown) => (typeof value === 'number' ? value : Number(value ?? 0));
  const percent = (value: number) => Math.max(0, Math.min(100, value * 100));
  const summary = $derived({
    correct: number(result.correct),
    total: number(result.total),
    accuracy: number(result.accuracy)
  });

  function breakdown(key: 'category' | 'tier'): Breakdown[] {
    const groups = result[key];
    if (!groups || typeof groups !== 'object' || Array.isArray(groups)) return [];
    return Object.entries(groups as Record<string, Stats>).map(([name, stats]) => ({
      name,
      correct: number(stats.correct),
      total: number(stats.total),
      accuracy: number(stats.accuracy)
    }));
  }

  const categoryRows = $derived(breakdown('category'));
  const tierRows = $derived(breakdown('tier'));
</script>

<section class="insights" aria-labelledby="puzzle-insights-title">
  <div class="insights-heading">
    <div><span class="eyebrow">Puzzle summary</span><h3 id="puzzle-insights-title">Where the model lands</h3></div>
    <span class="coverage">{summary.total} puzzles scored</span>
  </div>

  <div class="accuracy-card">
    <div class="ring" role="img" aria-label={`${(summary.accuracy * 100).toFixed(1)} percent accuracy`}>
      <svg viewBox="0 0 104 104" aria-hidden="true">
        <circle class="ring-track" cx="52" cy="52" r="43" pathLength="100" />
        <circle class="ring-value" cx="52" cy="52" r="43" pathLength="100" stroke-dasharray={`${percent(summary.accuracy)} 100`} />
      </svg>
      <strong>{(summary.accuracy * 100).toFixed(1)}<small>%</small></strong>
    </div>
    <div><strong class="score">{summary.correct} <span>/ {summary.total}</span></strong><p>correct moves</p><div class="progress-track" aria-hidden="true"><span style={`width: ${percent(summary.accuracy)}%`}></span></div></div>
  </div>

  {#if categoryRows.length || tierRows.length}
    <div class="breakdowns">
      {#each [{ label: 'Category', rows: categoryRows }, { label: 'Tier', rows: tierRows }] as group (group.label)}
        {#if group.rows.length}
          <div class="breakdown"><h4>{group.label}</h4>{#each group.rows as row (row.name)}
            <div class="bar-row"><div class="bar-label"><span>{row.name}</span><small>{row.correct}/{row.total} · {(row.accuracy * 100).toFixed(0)}%</small></div><div class="bar-track" role="progressbar" aria-label={`${group.label} ${row.name} accuracy`} aria-valuenow={percent(row.accuracy)} aria-valuemin="0" aria-valuemax="100"><span style={`width: ${percent(row.accuracy)}%`}></span></div></div>
          {/each}</div>
        {/if}
      {/each}
    </div>
  {/if}
</section>

<style>
  .insights{margin-top:20px;padding-top:20px;border-top:1px solid var(--line)}
  .insights-heading{display:flex;justify-content:space-between;align-items:end;gap:12px;margin-bottom:14px}.insights h3{margin:5px 0 0;font-size:15px}.coverage{color:var(--muted);font-size:10px}
  .accuracy-card{display:grid;grid-template-columns:92px 1fr;align-items:center;gap:16px;padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--panel-2)}
  .ring{position:relative;width:84px;height:84px}.ring svg{display:block;width:100%;height:100%;transform:rotate(-90deg)}.ring circle{fill:none;stroke-width:8}.ring-track{stroke:var(--line)}.ring-value{stroke:var(--accent);stroke-linecap:round}.ring strong{position:absolute;inset:0;display:grid;place-content:center;text-align:center;font-size:18px}.ring strong small{font-size:10px}.score{font-size:20px;font-variant-numeric:tabular-nums}.score span{color:var(--muted);font-size:14px;font-weight:500}.accuracy-card p{margin:4px 0 11px;color:var(--muted);font-size:11px}.progress-track,.bar-track{height:7px;overflow:hidden;border-radius:999px;background:var(--line)}.progress-track span,.bar-track span{display:block;height:100%;border-radius:inherit;background:var(--accent)}
  .breakdowns{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:18px}.breakdown h4{margin:0 0 11px;color:var(--muted);font-size:10px;text-transform:uppercase}.bar-row+.bar-row{margin-top:11px}.bar-label{display:flex;justify-content:space-between;gap:10px;margin-bottom:5px;font-size:11px}.bar-label small{color:var(--muted);font-size:10px;white-space:nowrap}.bar-track{height:6px}
  @media(max-width:560px){.breakdowns{grid-template-columns:1fr}}
</style>
