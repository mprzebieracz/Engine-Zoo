<script lang="ts">
  import type { Analysis, PolicyEntry } from '$lib/types';

  let { title = 'Agent output', subtitle = '', analysis, loading = false } = $props<{
    title?: string;
    subtitle?: string;
    analysis?: Analysis;
    loading?: boolean;
  }>();

  function rowsFrom(value: unknown): PolicyEntry[] {
    if (Array.isArray(value)) return value as PolicyEntry[];
    if (value && typeof value === 'object') return Object.entries(value as Record<string, number>).map(([move, p]) => ({ move, p }));
    return [];
  }

  const networkRows = $derived(topRows(rowsFrom(analysis?.network_policy ?? analysis?.policy)));
  const mctsRows = $derived(topRows(rowsFrom(analysis?.mcts_policy ?? analysis?.root_policy ?? analysis?.policy)));
  const displayedValue = $derived(typeof analysis?.network_value === 'number' ? analysis.network_value : analysis?.value);

  function probability(row: PolicyEntry, rows: PolicyEntry[]): number {
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
    return [...rows].sort((a, b) => probability(b, rows) - probability(a, rows)).slice(0, 5);
  }

  function label(row: PolicyEntry): string { return row.display_move ?? row.move ?? row.mv ?? '—'; }
</script>

<section class="panel">
  <header>
    <div><small>{subtitle}</small><h3>{title}</h3></div>
    <div class="summary"><span>{analysis?.best_move_san ?? analysis?.best_move ?? '—'}</span><strong>{typeof displayedValue === 'number' ? displayedValue.toFixed(3) : '—'}</strong></div>
  </header>

  {#if loading && !analysis}
    <div class="skeleton"></div><div class="skeleton short"></div>
  {:else if !analysis}
    <p class="empty">No cached output for this position yet.</p>
  {:else}
    <div class="policies">
      {@render PolicyColumn('Network', networkRows)}
      {@render PolicyColumn('MCTS', mctsRows)}
    </div>
  {/if}
</section>

{#snippet PolicyColumn(name: string, rows: PolicyEntry[])}
  <section class="policy-column">
    <div class="section-title"><span>{name}</span><small>policy</small></div>
    {#each rows as row (row.move ?? row.mv)}
      <div class="policy-row"><code>{label(row)}</code><div class="bar"><i style={`width:${Math.max(2, probability(row, rows) * 100)}%`}></i></div><span>{(probability(row, rows) * 100).toFixed(1)}%</span></div>
    {/each}
    {#if !rows.length}<p class="missing">Not returned.</p>{/if}
  </section>
{/snippet}

<style>
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:15px;min-height:0;overflow:hidden}header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px}header small{display:block;color:var(--muted);font-size:9px;text-transform:uppercase}h3{font-size:15px;margin:3px 0 0}.summary{display:flex;gap:12px;align-items:baseline;font-variant-numeric:tabular-nums}.summary span{font-size:13px}.summary strong{color:var(--accent);font-size:13px}
  .policies{display:grid;grid-template-columns:1fr 1fr;gap:14px}.section-title{display:flex;justify-content:space-between;padding-bottom:6px;border-bottom:1px solid var(--line);font-size:10px;font-weight:700}.section-title small{color:var(--muted);font-weight:400}.policy-row{display:grid;grid-template-columns:48px 1fr 35px;gap:6px;align-items:center;margin:7px 0;font-size:9px}.policy-row code{font-size:9px}.policy-row>span{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}.bar{height:4px;background:var(--panel-2);border-radius:99px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}.empty,.missing{color:var(--muted);font-size:11px}.empty{margin:20px 0}.missing{margin:10px 0}.skeleton{height:62px;border-radius:10px;background:var(--panel-2);margin-top:8px}.skeleton.short{height:38px}
  @media(max-width:520px){.policies{grid-template-columns:1fr}}
</style>
