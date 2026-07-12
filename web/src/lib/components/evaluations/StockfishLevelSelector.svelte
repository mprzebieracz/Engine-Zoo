<script lang="ts">
  type Level = { name: string; nodes: number; detail: string };

  let { value, onChange }: { value: number; onChange: (value: number) => void } = $props();

  const levels: Level[] = [
    { name: 'Scout', nodes: 1_000, detail: 'Fast signal' },
    { name: 'Club', nodes: 3_000, detail: 'Quick baseline' },
    { name: 'Tournament', nodes: 10_000, detail: 'Balanced' },
    { name: 'Serious', nodes: 30_000, detail: 'More confidence' },
    { name: 'Deep', nodes: 100_000, detail: 'Highest confidence' }
  ];

  const formatNodes = (nodes: number) => (nodes >= 1000 ? `${nodes / 1000}k` : `${nodes}`);
</script>

<label class="field">
  <span>Stockfish level</span>
  <select aria-describedby="stockfish-level-help" value={value} onchange={(event) => onChange(Number(event.currentTarget.value))}>
    {#each levels as level (level.nodes)}
      <option value={level.nodes}>{level.name} · {formatNodes(level.nodes)} nodes</option>
    {/each}
  </select>
  <small id="stockfish-level-help">Fixed budget: {formatNodes(value)} nodes per move.</small>
</label>

<style>
  .field{display:grid;gap:8px;margin-top:16px;color:var(--muted);font-size:12px;font-weight:650}
  .field select{width:100%;padding:12px 13px;border:1px solid var(--line);border-radius:11px;background:var(--panel-2);color:var(--text);font-weight:600}
  .field small{font-size:11px;font-weight:400;color:var(--muted)}
</style>
