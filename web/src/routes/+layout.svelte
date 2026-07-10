<script lang="ts">
  import '../app.css';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  let { children } = $props();
</script>

<header class="topbar">
  <a class="brand" href={resolve('/')}><span>EZ</span><div><strong>Engine Zoo</strong><small>Board game engine laboratory</small></div></a>
  <nav>
    <a class:active={page.url.pathname === '/'} href={resolve('/')}>Play</a>
    <a class:active={page.url.pathname.startsWith('/evaluate')} href={resolve('/evaluate')}>Evaluate</a>
  </nav>
  <div class="status"><i></i> local workspace</div>
</header>
{@render children()}

<style>
  .topbar{height:64px;padding:0 max(24px,calc((100vw - 1320px)/2));display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);background:var(--bg);position:sticky;top:0;z-index:50}.brand{display:flex;gap:12px;align-items:center;color:var(--text);text-decoration:none}.brand>span{width:38px;height:38px;border-radius:11px;display:grid;place-items:center;background:var(--accent);color:#11170b;font-weight:900}.brand strong,.brand small{display:block}.brand small{color:var(--muted);font-size:11px;margin-top:2px}nav{display:flex;padding:4px;border:1px solid var(--line);background:var(--panel);border-radius:12px}nav a{color:var(--muted);padding:8px 18px;border-radius:8px;text-decoration:none;font-size:13px}nav a.active{background:var(--panel-2);color:var(--text)}.status{color:var(--muted);font-size:11px;display:flex;align-items:center;gap:8px}.status i{width:7px;height:7px;border-radius:50%;background:var(--accent)}@media(max-width:700px){.topbar{padding:0 16px}.brand small,.status{display:none}nav a{padding:8px 12px}}
</style>
