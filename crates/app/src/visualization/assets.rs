pub(super) fn play_js() -> &'static str {
    r#"
let game = new Chess();
let board = null;
let uciHistory = [];
let orientation = 'white';
let thinking = false;
let autoplay = false;
let lastEngine = null;
let currentSeq = 0;

const $ = id => document.getElementById(id);
const sideName = color => color === 'w' ? 'White' : 'Black';
const playerFor = color => $(color === 'w' ? 'white-player' : 'black-player').value;
const modelFor = color => ($(color === 'w' ? 'white-model' : 'black-model').value || 'best').trim();
const simulations = () => Math.max(1, Number($('simulations').value) || 1);
const bothEngines = () => playerFor('w') === 'model' && playerFor('b') === 'model';
const isEngineTurn = () => !game.game_over() && playerFor(game.turn()) === 'model';

function moveToUci(move) {
  return move.from + move.to + (move.promotion || '');
}

function uciToMove(uci) {
  return {
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    promotion: uci.length > 4 ? uci.slice(4, 5) : undefined,
  };
}

function esc(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[char]));
}

function sanForUci(fen, uci) {
  if (!uci) return 'none';
  const preview = new Chess(fen);
  const move = preview.move(uciToMove(uci));
  return move ? move.san : uci;
}

async function api(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  const json = await response.json();
  if (!response.ok) throw new Error(json.error || response.statusText);
  return json;
}

function statusText() {
  if (game.in_checkmate()) return 'Checkmate';
  if (game.in_draw()) return 'Draw';
  if (game.in_check()) return sideName(game.turn()) + ' is in check';
  if (thinking) return sideName(game.turn()) + ' engine thinking';
  if (isEngineTurn()) return bothEngines() && !autoplay ? 'Engine ready' : 'Engine to move';
  return 'User to move';
}

function render() {
  if (board) {
    board.position(game.fen(), false);
    board.orientation(orientation);
  }
  $('turn').textContent = game.game_over() ? 'Game over' : sideName(game.turn()) + ' to move';
  $('status').textContent = statusText();
  $('fen').textContent = game.fen();
  $('pgn').textContent = game.pgn({ newline_char: ' ' }) || '-';
  renderLegal();
  renderLastEngine();
}

function renderLegal() {
  const moves = game.moves({ verbose: true });
  $('legal-count').textContent = moves.length + ' legal';
  $('legal').innerHTML = '';
  for (const move of moves) {
    const button = document.createElement('button');
    button.className = 'move-pill rounded-lg border border-line bg-paper px-3 py-2 text-sm font-semibold text-stone-800 hover:border-forest hover:bg-white';
    button.textContent = move.san;
    button.title = moveToUci(move);
    button.onclick = () => playMove({ from: move.from, to: move.to, promotion: move.promotion });
    $('legal').appendChild(button);
  }
}

function analysisCard(title, subtitle, analysis, fen) {
  const best = sanForUci(fen, analysis.best_move);
  const rows = (analysis.policy || [])
    .slice()
    .sort((a, b) => b.p - a.p)
    .slice(0, 12)
    .map(move => policyRow(move, fen))
    .join('');
  return `<div class="rounded-lg bg-paper p-3">
    <div class="text-xs font-semibold uppercase tracking-wide text-stone-500">${esc(subtitle)}</div>
    <div class="mt-1 flex items-end justify-between gap-3">
      <div class="text-2xl font-bold" title="${esc(analysis.best_move || '')}">${esc(best)}</div>
      <div class="text-sm font-semibold text-stone-700">value ${Number(analysis.value).toFixed(3)}</div>
    </div>
    <div class="mt-1 text-sm text-stone-600">${esc(title)}</div>
  </div>
  <div class="space-y-2">${rows || '<p class="text-stone-500">No legal policy.</p>'}</div>`;
}

function policyRow(move, fen) {
  const p = Math.max(0, Math.min(1, Number(move.p) || 0));
  const san = sanForUci(fen, move.mv);
  return `<div class="grid grid-cols-[70px_1fr_52px] items-center gap-3">
    <span class="font-mono text-stone-700" title="${esc(move.mv)}">${esc(san)}</span>
    <span class="policy-track h-2 overflow-hidden rounded-full">
      <span class="policy-fill block h-full rounded-full" style="width:${Math.max(2, p * 100)}%"></span>
    </span>
    <strong class="text-right text-stone-700">${(p * 100).toFixed(1)}%</strong>
  </div>`;
}

function renderLastEngine() {
  if (!lastEngine) {
    $('engine-analysis').innerHTML = '<p class="text-stone-500">No engine move yet.</p>';
    return;
  }
  $('engine-analysis').innerHTML = analysisCard(
    `${lastEngine.side} ${lastEngine.model} played ${lastEngine.san} (${lastEngine.uci})`,
    'MCTS policy before move',
    lastEngine.analysis,
    lastEngine.fen,
  );
}

function requestFor(color, mode) {
  return {
    position: { game: 'chess', position: { moves: uciHistory.slice() } },
    model: modelFor(color),
    mode,
    simulations: simulations(),
  };
}

function normalizeMove(input) {
  const trimmed = input.trim();
  if (!trimmed) return null;
  if (/^[a-h][1-8][a-h][1-8][qrbn]?$/i.test(trimmed)) return uciToMove(trimmed.toLowerCase());
  const move = game.move(trimmed, { sloppy: true });
  if (!move) return null;
  game.undo();
  return { from: move.from, to: move.to, promotion: move.promotion };
}

function playMove(move) {
  if (thinking) return false;
  const made = game.move(move);
  if (!made) return false;
  uciHistory.push(moveToUci(made));
  render();
  queueEngine();
  return true;
}

function onDragStart(source, piece) {
  if (thinking || game.game_over() || isEngineTurn()) return false;
  if (game.turn() === 'w' && piece.startsWith('b')) return false;
  if (game.turn() === 'b' && piece.startsWith('w')) return false;
  return true;
}

function onDrop(source, target) {
  const move = { from: source, to: target };
  const piece = game.get(source);
  if (piece && piece.type === 'p' && (target[1] === '8' || target[1] === '1')) {
    move.promotion = (prompt('Promote to q, r, b, or n', 'q') || 'q').toLowerCase()[0];
  }
  return playMove(move) ? undefined : 'snapback';
}

async function queueEngine() {
  if (game.game_over() || !isEngineTurn()) return;
  if (bothEngines() && !autoplay) return;
  await engineMove();
}

async function engineMove() {
  if (thinking || game.game_over() || !isEngineTurn()) return;
  const color = game.turn();
  const model = modelFor(color);
  const beforeFen = game.fen();
  thinking = true;
  render();
  try {
    const analysis = await api('/analyze', requestFor(color, 'mcts'));
    const uci = analysis.best_move;
    if (!uci) throw new Error('engine returned no move');
    const made = game.move(uciToMove(uci));
    if (!made) throw new Error('engine returned illegal move ' + uci);
    uciHistory.push(moveToUci(made));
    lastEngine = { side: sideName(color), model, uci, fen: beforeFen, san: made.san, analysis };
    thinking = false;
    render();
    if (autoplay) setTimeout(queueEngine, 120);
  }
  catch (error) {
    thinking = false;
    autoplay = false;
    $('autoplay').textContent = 'Start Engines';
    $('status').textContent = error.message;
    render();
  }
}

function undoOne() {
  if (thinking) return;
  const move = game.undo();
  if (move) uciHistory.pop();
  lastEngine = null;
  render();
}

async function analyzeCurrent() {
  const seq = ++currentSeq;
  const fen = game.fen();
  $('current-analysis').innerHTML = '<p class="text-stone-500">Analyzing current position...</p>';
  try {
    const color = game.turn();
    const analysis = await api('/analyze', requestFor(color, 'net'));
    if (seq === currentSeq) {
      $('current-analysis').innerHTML = analysisCard(
        `${sideName(color)} to move with ${modelFor(color)}`,
        'Raw network policy',
        analysis,
        fen,
      );
    }
  }
  catch (error) {
    if (seq === currentSeq) $('current-analysis').innerHTML = `<p class="text-danger">${esc(error.message)}</p>`;
  }
}

function newGame() {
  game = new Chess();
  uciHistory = [];
  lastEngine = null;
  autoplay = false;
  $('autoplay').textContent = 'Start Engines';
  $('current-analysis').innerHTML = '<p class="text-stone-500">Press Analyze for the current board.</p>';
  render();
  queueEngine();
}

$('new-game').onclick = newGame;
$('flip').onclick = () => { orientation = orientation === 'white' ? 'black' : 'white'; render(); };
$('undo').onclick = undoOne;
$('undo-pair').onclick = () => { undoOne(); undoOne(); };
$('step-engine').onclick = engineMove;
$('autoplay').onclick = () => {
  autoplay = !autoplay;
  $('autoplay').textContent = autoplay ? 'Stop Engines' : 'Start Engines';
  render();
  if (autoplay) queueEngine();
};
$('analyze-current').onclick = analyzeCurrent;
$('white-player').onchange = () => { render(); queueEngine(); };
$('black-player').onchange = () => { render(); queueEngine(); };
$('move-form').onsubmit = event => {
  event.preventDefault();
  const move = normalizeMove($('move').value);
  if (move && playMove(move)) $('move').value = '';
};

board = Chessboard('board', {
  position: game.fen(),
  draggable: true,
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
  onDragStart,
  onDrop,
  onSnapEnd: render,
});
window.addEventListener('resize', () => board && board.resize());
render();
queueEngine();
"#
}

pub(super) fn report_css() -> &'static str {
    r#":root{color-scheme:dark;--bg:#111315;--panel:#1b2024;--panel2:#22282d;--line:#343c43;--text:#f0f3f5;--muted:#9da7af;--ok:#57b98f;--bad:#e06a64;--accent:#d9a441;--bar:#395f50}*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#151819,#0f1112);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}.report{max-width:1240px;margin:0 auto;padding:28px}.report-header{margin-bottom:18px}h1{margin:0;font-size:32px;letter-spacing:0}.muted{color:var(--muted)}code{color:#d6dee5}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}.summary div,.case,.toolbar{background:var(--panel);border:1px solid var(--line);border-radius:8px}.summary div{padding:16px}.summary span,.facts span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase}.summary strong{font-size:30px}.toolbar{display:grid;grid-template-columns:2fr 180px 260px;gap:10px;padding:12px;margin-bottom:12px;position:sticky;top:0;z-index:2}input,select{width:100%;background:#252b30;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:9px 11px}.cases{display:grid;gap:10px}.case-head{width:100%;display:grid;grid-template-columns:1fr 110px auto;gap:12px;align-items:center;text-align:left;background:var(--panel);color:var(--text);border:0;padding:14px 16px;cursor:pointer}.case-title{display:grid;gap:3px}.case-title strong{font-size:16px}.case-title small{color:var(--accent)}.best{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:800}.best.ok{color:var(--ok)}.best.bad{color:var(--bad)}.badge{border-radius:999px;padding:4px 9px;font-weight:800;text-transform:uppercase;font-size:12px}.badge.pass{background:rgba(79,178,134,.16);color:var(--ok)}.badge.fail{background:rgba(217,93,93,.16);color:var(--bad)}.case-body{display:none;grid-template-columns:380px 1fr;gap:18px;padding:16px;border-top:1px solid var(--line);background:var(--panel2)}.case.open .case-body{display:grid}.chessboard{display:grid;grid-template-columns:repeat(8,1fr);aspect-ratio:1;border:1px solid var(--line);border-radius:6px;overflow:hidden}.sq{display:flex;align-items:center;justify-content:center;font-size:30px}.light{background:#d6c9a8;color:#111}.dark{background:#718568;color:#111}pre{white-space:pre-wrap;color:var(--muted);font-size:12px}.facts{display:grid;align-content:start;gap:12px}.facts strong{font-size:18px}.facts h3{margin:8px 0 0}.policy-row{display:grid;grid-template-columns:78px 1fr 58px;gap:10px;align-items:center}.policy-row span{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--text)}.bar{height:10px;background:#313941;border-radius:999px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--bar),var(--accent));border-radius:999px}@media(max-width:860px){.summary,.case-body,.toolbar{grid-template-columns:1fr}.case-head{grid-template-columns:1fr auto}.best{display:none}}"#
}

pub(super) fn report_js() -> &'static str {
    r#"document.querySelectorAll('.case-head').forEach(b=>b.onclick=()=>b.closest('.case').classList.toggle('open'));const filter=document.getElementById('filter'),status=document.getElementById('status-filter'),category=document.getElementById('category-filter');function apply(){const q=filter.value.toLowerCase(),s=status.value,k=category.value;document.querySelectorAll('.case').forEach(c=>{const ok=(!q||c.dataset.text.toLowerCase().includes(q))&&(s==='all'||c.dataset.status===s)&&(k==='all'||c.dataset.category===k);c.style.display=ok?'':'none'})}filter.oninput=apply;status.onchange=apply;category.onchange=apply;"#
}
