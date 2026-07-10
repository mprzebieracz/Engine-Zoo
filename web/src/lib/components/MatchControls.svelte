<script lang="ts">
  let { ply, total, autoplay, pace, automated, terminal, onprevious, onnext, onlive, ontoggle, onpace } = $props<{
    ply: number;
    total: number;
    autoplay: boolean;
    pace: number;
    automated: boolean;
    terminal: boolean;
    onprevious: () => void;
    onnext: () => void;
    onlive: () => void;
    ontoggle: () => void;
    onpace: (pace: number) => void;
  }>();
</script>

<div class="controls" aria-label="Match playback">
  <div class="navigation">
    <button onclick={onprevious} disabled={ply === 0} aria-label="Previous position">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m15 5-7 7 7 7" /></svg>
    </button>
    <button onclick={onnext} disabled={ply === total} aria-label="Next position">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7" /></svg>
    </button>
    <button class="live" class:active={ply === total} onclick={onlive}>Live <span>{ply}/{total}</span></button>
  </div>
  {#if automated}
    <div class="playback">
      <button class="play" onclick={ontoggle} disabled={terminal} aria-label={autoplay ? 'Pause match' : 'Continue match'}>{autoplay ? 'Pause' : 'Continue'}</button>
      <label><span>Pace</span><select value={pace} onchange={(event) => onpace(Number(event.currentTarget.value))}><option value="200">Fast</option><option value="700">Normal</option><option value="1500">Slow</option><option value="3000">Review</option></select></label>
    </div>
  {/if}
</div>

<style>
  .controls{width:100%;display:flex;align-items:center;justify-content:space-between;gap:12px}.navigation,.playback{display:flex;align-items:center;gap:8px}button,select{height:36px;border:1px solid var(--line);border-radius:9px;background:var(--panel);color:var(--text);padding:0 12px;cursor:pointer}button:disabled{opacity:.4;cursor:not-allowed}.navigation>button:not(.live){width:48px;height:44px;padding:0;display:grid;place-items:center}.navigation>button:not(.live) svg{width:24px;height:24px;fill:none;stroke:currentColor;stroke-linecap:round;stroke-linejoin:round;stroke-width:2}.live{height:44px;display:flex;gap:8px;align-items:center}.live span{color:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}.live.active{border-color:var(--accent)}.play{min-width:80px}.playback label{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:10px}.playback select{min-width:90px}@media(max-width:600px){.controls{align-items:stretch;flex-direction:column}.navigation,.playback{justify-content:space-between}.navigation .live{flex:1;justify-content:center}}
</style>
