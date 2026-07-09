<script lang="ts">
  import type { Analysis, PolicyEntry } from '$lib/types';
  let { analysis, loading = false } = $props<{ analysis?: Analysis; loading?: boolean }>();

  function rowsFrom(value: unknown): PolicyEntry[] {
    if (Array.isArray(value)) return value as PolicyEntry[];
    if (value && typeof value === 'object') {
      return Object.entries(value as Record<string, number>).map(([move, probability]) => ({ move, probability }));
    }
    return [];
  }

  function nested(...keys: string[]): unknown {
    let value: unknown = analysis;
    for (const key of keys) {
      if (!value || typeof value !== 'object') return undefined;
      value = (value as Record<string, unknown>)[key];
    }
    return value;
  }

  const networkRows = $derived(topRows(rowsFrom(analysis?.network_policy ?? nested('network', 'policy') ?? analysis?.policy)));
  const mctsRows = $derived(topRows(rowsFrom(analysis?.mcts_policy ?? analysis?.root_policy ?? nested('mcts', 'policy') ?? analysis?.moves)));
  const displayedValue = $derived(
    typeof analysis?.value === 'number' ? analysis.value :
    typeof nested('network', 'value') === 'number' ? nested('network', 'value') as number : undefined
  );

  function weight(row: PolicyEntry, rows: PolicyEntry[]): number {
    if (typeof row.p === 'number') return row.p;
    if (typeof row.probability === 'number') return row.probability;
    if (typeof row.prior === 'number') return row.prior;
    const visits = row.visits ?? row.visit_count;
    if (typeof visits === 'number') {
      const total = rows.reduce((sum, entry) => sum + (entry.visits ?? entry.visit_count ?? 0), 0);
      return total ? visits / total : 0;
    }
    return 0;
  }

  function topRows(rows: PolicyEntry[]): PolicyEntry[] {
    return [...rows].sort((a, b) => weight(b, rows) - weight(a, rows)).slice(0, 8);
  }

  function label(row: PolicyEntry): string {
    return row.display_move ?? row.move ?? row.mv ?? '—';
  }
</script>

<section class="panel">
  <header>
    <div><span class="eyebrow">Live inspection</span><h3>Agent output</h3></div>
    <span class:working={loading} class="status">{loading ? 'thinking' : 'ready'}</span>
  </header>

  {#if !analysis && !loading}
    <div class="empty">Play a move or request analysis to inspect the network and search distributions.</div>
  {:else if loading}
    <div class="skeleton value"></div><div class="skeleton"></div><div class="skeleton"></div>
  {:else}
    <div class="metrics">
      <div><small>Best move</small><strong>{analysis?.best_move_san ?? analysis?.best_move ?? '—'}</strong></div>
      <div class="value-card">
        <small>Position value</small>
        <strong>{typeof displayedValue === 'number' ? displayedValue.toFixed(3) : '—'}</strong>
        {#if typeof displayedValue === 'number'}
          <div class="value-track"><i style={`left:${Math.max(0, Math.min(100, (displayedValue + 1) * 50))}%`}></i></div>
          <span>−1 loss <b>0</b> +1 win</span>
        {/if}
      </div>
    </div>

    <div class="tabs-grid">
      <section>
        <div class="section-title"><span>Network policy</span><small>prior</small></div>
        {#if networkRows.length}
          {#each networkRows as row, i}
            <div class="policy-row">
              <span class="rank">{i + 1}</span><code>{label(row)}</code>
              <div class="bar prior"><i style={`width:${Math.max(1.5, weight(row, networkRows) * 100)}%`}></i></div>
              <span>{(weight(row, networkRows) * 100).toFixed(1)}%</span>
            </div>
          {/each}
        {:else}<p class="missing">Not present in this response.</p>{/if}
      </section>

      <section>
        <div class="section-title"><span>MCTS policy</span><small>visits</small></div>
        {#if mctsRows.length}
          {#each mctsRows as row, i}
            <div class="policy-row">
              <span class="rank">{i + 1}</span><code>{label(row)}</code>
              <div class="bar"><i style={`width:${Math.max(1.5, weight(row, mctsRows) * 100)}%`}></i></div>
              <span>{(weight(row, mctsRows) * 100).toFixed(1)}%</span>
            </div>
          {/each}
        {:else}<p class="missing">Not present in this response.</p>{/if}
      </section>
    </div>

    <details><summary>Raw response</summary><pre>{JSON.stringify(analysis, null, 2)}</pre></details>
  {/if}
</section>

<style>
  .panel { background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:20px;min-height:330px; }
  header { display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px; }
  h3 { margin:3px 0 0;font-size:18px; }.eyebrow { color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.12em; }
  .status { font-size:10px;padding:6px 9px;border-radius:99px;color:var(--accent);background:color-mix(in srgb,var(--accent) 10%,transparent);text-transform:uppercase;letter-spacing:.08em; }
  .working { animation:pulse 1s infinite alternate; }
  .metrics { display:grid;grid-template-columns:.72fr 1.28fr;gap:10px;margin-bottom:18px; }
  .metrics>div { background:var(--panel-2);border-radius:14px;padding:14px; }.metrics small { display:block;color:var(--muted);margin-bottom:6px;font-size:10px;text-transform:uppercase;letter-spacing:.08em; }.metrics strong{font-size:19px;}
  .value-track { height:5px;background:linear-gradient(90deg,#ef6d78 0 50%,var(--accent) 50% 100%);border-radius:99px;margin:11px 0 5px;position:relative; }.value-track i{position:absolute;width:2px;height:11px;top:-3px;background:white;border-radius:2px;}.value-card>span{display:flex;justify-content:space-between;color:var(--muted);font-size:8px;}.value-card b{font-weight:400;}
  .tabs-grid { display:grid;grid-template-columns:1fr 1fr;gap:16px; }.section-title{display:flex;justify-content:space-between;padding-bottom:8px;border-bottom:1px solid var(--line);font-size:11px;font-weight:700;}.section-title small{color:var(--muted);font-weight:400;}
  .policy-row { display:grid;grid-template-columns:16px 54px 1fr 39px;gap:7px;align-items:center;font-size:10px;margin:9px 0; }.rank{color:var(--muted)}code{font-size:10px}.bar{height:5px;background:var(--panel-2);border-radius:99px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent);border-radius:inherit}.bar.prior i{background:#70a7ff}.policy-row>span:last-child{text-align:right;color:var(--muted)}
  .missing,.empty{color:var(--muted);font-size:12px;line-height:1.6}.empty{padding:36px 8px}.missing{padding:12px 0}
  details{margin-top:16px;color:var(--muted);font-size:11px}summary{cursor:pointer}pre{max-height:260px;overflow:auto;background:#080a0e;border-radius:10px;padding:12px;font-size:10px}
  .skeleton{height:76px;border-radius:12px;margin:10px 0;background:linear-gradient(90deg,var(--panel-2),#252b37,var(--panel-2));background-size:220% 100%;animation:shimmer 1.2s infinite}.skeleton.value{height:92px}
  @keyframes pulse{to{opacity:.4}}@keyframes shimmer{to{background-position:-220% 0}}
  @media(max-width:550px){.tabs-grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}}
</style>
