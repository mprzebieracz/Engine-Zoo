<script lang="ts">
 import type { AgentDescriptor } from '$lib/types';
 let { agent, side, active } = $props<{agent?:AgentDescriptor;side:string;active:boolean}>();
</script>
<div class:active class="player-strip"><div class="avatar">{agent?.kind==='human'?'U':'α'}</div><div><strong>{agent?.name??'Agent'}</strong><small>{side}</small></div><i></i><span>{active?'to move':'waiting'}</span></div>
<style>.player-strip{width:100%;display:grid;grid-template-columns:38px 1fr 8px auto;gap:10px;align-items:center;padding:7px 9px;border-radius:11px;color:var(--muted)}.player-strip.active{background:var(--panel)}.avatar{width:34px;height:34px;border-radius:9px;display:grid;place-items:center;background:var(--panel-2);color:var(--accent);font-weight:800}.player-strip strong,.player-strip small{display:block}.player-strip strong{color:var(--text);font-size:12px}.player-strip small,.player-strip span{font-size:9px;text-transform:uppercase}.player-strip i{width:7px;height:7px;border-radius:50%;background:#414752}.player-strip.active i{background:var(--accent)}</style>
