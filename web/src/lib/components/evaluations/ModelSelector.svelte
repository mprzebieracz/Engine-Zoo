<script lang="ts">
  let { checkpoints, value, onChange }: { checkpoints: string[]; value: string; onChange: (value: string) => void } = $props();
</script>
<label class="field"><span>Candidate checkpoint</span><select value={value} onchange={(event) => onChange(event.currentTarget.value)} disabled={!checkpoints.length}>
  {#each checkpoints as checkpoint (checkpoint)}<option value={checkpoint}>{checkpoint}</option>{/each}
  {#if !checkpoints.length}<option>No checkpoints found</option>{/if}
</select></label>
<style>.field{display:grid;gap:8px;color:var(--muted);font-size:12px;font-weight:650}.field select{width:100%;padding:12px 13px;border:1px solid var(--line);border-radius:11px;background:var(--panel-2);color:var(--text);font-weight:600}</style>
