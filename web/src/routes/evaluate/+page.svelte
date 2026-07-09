<script lang="ts">
  import { goto } from '$app/navigation';
</script>
<svelte:head><title>Evaluate · Engine Zoo</title></svelte:head>
<main class="page screen-enter"><span class="eyebrow">Evaluation workspace</span><h1 class="hero-title">Challenge suites.</h1><p class="lead">The evaluation workflow remains available as a separate product surface. Add suite loading and persisted jobs here without mixing it into match setup.</p><section><h2>Puzzle evaluation</h2><p>Create JSONL suites, select an agent, run network or MCTS analysis, and inspect each case.</p><button class="primary" onclick={()=>goto('/')}>Create a match first</button></section></main>
<style>section{margin-top:42px;padding:28px;border:1px solid var(--line);border-radius:22px;background:var(--panel)}section h2{margin-top:0}section p{color:var(--muted);line-height:1.6;margin-bottom:24px}</style>
