<script lang="ts">
  import { onMount } from 'svelte';
  import { evaluationsApi } from '$lib/api/evaluations';
  import type { Catalog, EvaluationJob, Suite } from '$lib/evaluations/types';
  import ModelSelector from '$lib/components/evaluations/ModelSelector.svelte';
  import SuiteCards from '$lib/components/evaluations/SuiteCards.svelte';
  import JobList from '$lib/components/evaluations/JobList.svelte';
  import ResultPanel from '$lib/components/evaluations/ResultPanel.svelte';
  import StockfishLevelSelector from '$lib/components/evaluations/StockfishLevelSelector.svelte';

  let catalog = $state<Catalog | undefined>();
  let jobs = $state<EvaluationJob[]>([]);
  let selectedId = $state('');
  let candidate = $state('');
  let baseline = $state('');
  let suite = $state<Suite>('puzzle');
  let simulations = $state(400);
  let rounds = $state(2);
  let stockfishNodes = $state(3000);
  let busy = $state(false);
  let error = $state('');
  let selected = $derived(jobs.find((job) => job.id === selectedId));
  let preflight = $derived(catalog?.preflight ?? { puzzle: { available: false, message: 'Loading…' }, stockfish: { available: false, message: 'Loading…' }, arena: { available: false, message: 'Loading…' } });

  onMount(() => {
    void load();

    const timer = window.setInterval(refreshActiveJobs, 2000);
    return () => window.clearInterval(timer);
  });

  async function load() {
    try {
      [catalog, jobs] = await Promise.all([evaluationsApi.catalog(), evaluationsApi.jobs()]);
      setInitialSelection();
    } catch (cause) {
      error = messageFor(cause, 'Could not load evaluations');
    }
  }

  function setInitialSelection() {
    if (!catalog) return;

    candidate = catalog.checkpoints[0] ?? '';
    baseline = catalog.checkpoints[1] ?? candidate;
    selectedId = jobs[0]?.id ?? '';
  }

  function refreshActiveJobs() {
    if (jobs.some((job) => job.status === 'queued' || job.status === 'running')) {
      void refreshJobs();
    }
  }

  async function refreshJobs() {
    try {
      jobs = await evaluationsApi.jobs();
    } catch {
      // Retain the last persisted view.
    }
  }

  function chooseSuite(next: Suite) {
    suite = next;

    const defaults = catalog?.suites.find((item) => item.id === next)?.defaults;
    if (!defaults) return;

    if (defaults.simulations) simulations = defaults.simulations;
    if (defaults.rounds) rounds = defaults.rounds;
    if (defaults.stockfish_nodes) stockfishNodes = defaults.stockfish_nodes;
  }

  async function run() {
    error = '';
    busy = true;

    try {
      const job = await evaluationsApi.create({
        candidate,
        suite,
        baseline: suite === 'arena' ? baseline : undefined,
        simulations,
        rounds,
        stockfish_nodes: stockfishNodes
      });

      jobs = [job, ...jobs];
      selectedId = job.id;
    } catch (cause) {
      error = messageFor(cause, 'Could not create evaluation');
    } finally {
      busy = false;
    }
  }

  function messageFor(cause: unknown, fallback: string): string {
    return cause instanceof Error ? cause.message : fallback;
  }
</script>

<svelte:head><title>Evaluate · Engine Zoo</title></svelte:head>
<main class="page evaluate-page">
  <header class="heading"><div><span class="eyebrow">Evaluation workspace</span><h1 class="hero-title">Measure the next move.</h1><p class="lead">Run approved local suites against saved chess checkpoints. Results persist beside the training run.</p></div><div class="server-state"><span class:online={!!catalog}>●</span>{catalog ? 'Server ready' : 'Connecting'}</div></header>
  {#if error}<div class="error">{error}</div>{/if}
  <div class="workspace"><section class="controls"><div class="panel"><h2>1. Select a model</h2><ModelSelector checkpoints={catalog?.checkpoints ?? []} value={candidate} onChange={(value) => candidate = value} />{#if candidate}<p class="detail">{candidate} · resolved from the configured chess run directory</p>{/if}</div>
    <div class="panel"><div class="section-title"><div><h2>2. Choose a suite</h2><p>Preflight checks come from the server.</p></div></div><SuiteCards value={suite} preflight={preflight} onChange={chooseSuite} />
      {#if suite === 'arena'}<ModelSelector checkpoints={(catalog?.checkpoints ?? []).filter((item) => item !== candidate)} value={baseline} onChange={(value) => baseline = value} />{:else if suite === 'stockfish'}<StockfishLevelSelector value={stockfishNodes} onChange={(value) => stockfishNodes = value} />{/if}
      <div class="settings"><label class="field"><span>Candidate simulations</span><input type="number" min="1" max="10000" bind:value={simulations} /></label>{#if suite !== 'puzzle'}<label class="field"><span>Opening rounds</span><input type="number" min="1" max="20" bind:value={rounds} /></label>{/if}</div>
      <div class="preflight"><span>Preflight</span><b class:good={preflight[suite]?.available}>{preflight[suite]?.message}</b></div><button class="primary run" disabled={busy || !candidate || !preflight[suite]?.available || (suite === 'arena' && !baseline)} onclick={run}>{busy ? 'Queueing…' : 'Run evaluation'}</button>
    </div>
  </section><aside class="history"><section><div class="section-title"><div><span class="eyebrow">Recent status</span><h2>History</h2></div>{#if jobs.length}<span class="count">{jobs.length}</span>{/if}</div><JobList {jobs} selected={selectedId} onSelect={(id) => selectedId = id} /></section><ResultPanel job={selected} /></aside></div>
</main>
<style>
  .evaluate-page{max-width:1380px}.heading{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}.server-state{padding:9px 12px;border:1px solid var(--line);border-radius:10px;color:var(--muted);font-size:11px}.server-state span{color:#ff9da7;margin-right:7px}.server-state span.online{color:var(--accent)}.workspace{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(360px,.85fr);gap:18px;margin-top:38px}.controls,.history{display:grid;align-content:start;gap:14px}.panel,.history section{padding:20px;border:1px solid var(--line);border-radius:16px;background:var(--panel)}h2{margin:0 0 17px;font-size:18px}.section-title{display:flex;justify-content:space-between;align-items:start;margin-bottom:17px}.section-title h2{margin:5px 0 0}.section-title p,.detail{margin:0;color:var(--muted);font-size:12px;line-height:1.5}.field{display:grid;gap:8px;margin-top:16px;color:var(--muted);font-size:12px;font-weight:650}.field input{width:100%;padding:11px 12px;border:1px solid var(--line);border-radius:10px;background:var(--panel-2);color:var(--text)}.settings{display:grid;grid-template-columns:1fr 1fr;gap:10px}.preflight{display:flex;justify-content:space-between;gap:12px;margin-top:20px;padding:10px 12px;background:var(--panel-2);border-radius:9px;color:var(--muted);font-size:11px}.preflight b{font-weight:600;color:#ff9da7;text-align:right}.preflight b.good{color:var(--accent)}.run{width:100%;margin-top:14px}.count{padding:4px 7px;border-radius:7px;background:var(--panel-2);color:var(--muted);font-size:11px}.error{margin-top:22px}.history>section{padding-bottom:16px}@media(max-width:900px){.workspace{grid-template-columns:1fr}.history{order:2}}@media(max-width:560px){.heading{display:block}.server-state{display:inline-block;margin-top:15px}.settings{grid-template-columns:1fr}}
</style>
