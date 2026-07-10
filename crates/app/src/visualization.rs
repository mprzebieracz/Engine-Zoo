use crate::proxy::{game_name, GameKind};
use algorithms::analysis::Analysis;
use games::position::{PositionGame, PositionSpec};
use games::ChessGame;

pub struct BenchReport<'a> {
    pub title: &'a str,
    pub game: GameKind,
    pub model: &'a str,
    pub mode: &'a str,
    pub rows: &'a [BenchReportRow],
}

pub struct BenchReportRow {
    pub name: Option<String>,
    pub category: Option<String>,
    pub position: PositionSpec,
    pub board: String,
    pub expected: Vec<String>,
    pub best_move: Option<String>,
    pub correct: bool,
    pub analysis: Analysis,
}

pub fn render_chess_play_page(game: GameKind, run_dir: &str) -> String {
    let title = format!("engine-zoo {}", game_name(game));
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{}</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {{ theme: {{ extend: {{ colors: {{ ink: '#1f2328', paper: '#f5f1e8', panel: '#fffdf8', line: '#ddd3c0', forest: '#2f654f', gold: '#b88a35', danger: '#a64038' }} }} }} }}
</script>
<link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<style>
body{{background:#f5f1e8}}.board-bounds .white-1e1d7{{background:#d8caa5}}.board-bounds .black-3c85d{{background:#78906b}}.move-pill{{transition:border-color .12s ease,background .12s ease,color .12s ease}}.policy-track{{background:#eadfca}}.policy-fill{{background:linear-gradient(90deg,#2f654f,#b88a35)}}.scroll-slim{{scrollbar-width:thin}}.scroll-slim::-webkit-scrollbar{{height:8px;width:8px}}.scroll-slim::-webkit-scrollbar-thumb{{background:#c8bda9;border-radius:999px}}
</style>
</head>
<body class="text-ink antialiased">
<main class="mx-auto max-w-[1500px] px-5 py-5">
  <header class="mb-6 flex flex-col gap-4 border-b border-line pb-5 lg:flex-row lg:items-end lg:justify-between">
    <div>
      <p class="text-sm font-semibold uppercase tracking-wide text-forest">engine-zoo chess</p>
      <h1 class="mt-1 text-3xl font-bold tracking-tight">{}</h1>
      <p class="mt-2 text-sm text-stone-600">Run directory <code class="rounded bg-white px-1.5 py-0.5 text-stone-800 shadow-sm">{}</code></p>
    </div>
    <div class="grid grid-cols-2 gap-3 md:grid-cols-6">
      <label class="text-sm"><span class="mb-1 block font-medium text-stone-600">White</span><select id="white-player" class="w-full rounded-lg border border-line bg-white px-3 py-2 shadow-sm outline-none focus:border-forest"><option value="human">User</option><option value="model">Engine</option></select></label>
      <label class="text-sm"><span class="mb-1 block font-medium text-stone-600">White model</span><input id="white-model" class="w-full rounded-lg border border-line bg-white px-3 py-2 shadow-sm outline-none focus:border-forest" value="best"></label>
      <label class="text-sm"><span class="mb-1 block font-medium text-stone-600">Black</span><select id="black-player" class="w-full rounded-lg border border-line bg-white px-3 py-2 shadow-sm outline-none focus:border-forest"><option value="model">Engine</option><option value="human">User</option></select></label>
      <label class="text-sm"><span class="mb-1 block font-medium text-stone-600">Black model</span><input id="black-model" class="w-full rounded-lg border border-line bg-white px-3 py-2 shadow-sm outline-none focus:border-forest" value="best"></label>
      <label class="text-sm"><span class="mb-1 block font-medium text-stone-600">Simulations</span><input id="simulations" class="w-full rounded-lg border border-line bg-white px-3 py-2 shadow-sm outline-none focus:border-forest" type="number" value="800" min="1"></label>
      <button id="new-game" class="self-end rounded-lg bg-forest px-4 py-2.5 font-semibold text-white shadow-sm hover:bg-[#28523f]">New Game</button>
    </div>
  </header>

  <section class="grid gap-5 xl:grid-cols-[minmax(500px,650px)_minmax(360px,1fr)_minmax(330px,420px)]">
    <section class="rounded-xl border border-line bg-panel p-5 shadow-sm">
      <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div id="turn" class="text-xl font-bold">White to move</div>
          <div id="status" class="mt-1 text-sm text-stone-600">Ready</div>
        </div>
        <div class="flex flex-wrap gap-2">
          <button id="flip" class="rounded-lg border border-line bg-white px-3 py-2 text-sm font-semibold hover:border-forest">Flip</button>
          <button id="undo" class="rounded-lg border border-line bg-white px-3 py-2 text-sm font-semibold hover:border-forest">Undo</button>
          <button id="undo-pair" class="rounded-lg border border-line bg-white px-3 py-2 text-sm font-semibold hover:border-forest">Undo 2</button>
        </div>
      </div>
      <div class="board-bounds mx-auto w-full max-w-[620px] overflow-hidden rounded-lg shadow-lg">
        <div id="board"></div>
      </div>
      <form id="move-form" class="mt-5 flex gap-3">
        <input id="move" class="min-w-0 flex-1 rounded-lg border border-line bg-paper px-3 py-2 outline-none focus:border-forest" placeholder="uci or san, e.g. e2e4 / Nf3" autocomplete="off">
        <button class="rounded-lg bg-forest px-4 py-2 font-semibold text-white hover:bg-[#28523f]">Play</button>
      </form>
    </section>

    <aside class="grid content-start gap-5">
      <section class="rounded-xl border border-line bg-panel p-5 shadow-sm">
        <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 class="text-xl font-bold">Game</h2>
          <div class="flex gap-2">
            <button id="step-engine" class="rounded-lg border border-forest px-3 py-2 text-sm font-semibold text-forest hover:bg-paper">Step Engine</button>
            <button id="autoplay" class="rounded-lg bg-forest px-3 py-2 text-sm font-semibold text-white hover:bg-[#28523f]">Start Engines</button>
          </div>
        </div>
        <div class="grid gap-3 text-sm">
          <div class="rounded-lg bg-paper p-3"><span class="block text-xs font-semibold uppercase tracking-wide text-stone-500">PGN</span><div id="pgn" class="scroll-slim mt-2 max-h-44 overflow-auto font-mono leading-7 text-stone-800">-</div></div>
          <div class="rounded-lg bg-paper p-3"><span class="block text-xs font-semibold uppercase tracking-wide text-stone-500">FEN</span><div id="fen" class="mt-2 break-all font-mono text-xs text-stone-700"></div></div>
        </div>
      </section>

      <section class="rounded-xl border border-line bg-panel p-5 shadow-sm">
        <div class="mb-4 flex items-center justify-between">
          <h2 class="text-xl font-bold">Legal Moves</h2>
          <span id="legal-count" class="rounded-full bg-paper px-3 py-1 text-xs font-semibold text-stone-600"></span>
        </div>
        <div id="legal" class="scroll-slim flex max-h-80 flex-wrap gap-2 overflow-auto pr-1"></div>
      </section>
    </aside>

    <aside class="grid content-start gap-5">
      <section class="rounded-xl border border-line bg-panel p-5 shadow-sm">
        <div class="mb-4 flex items-center justify-between gap-3">
          <div>
            <h2 class="text-xl font-bold">Last Engine Decision</h2>
            <p class="text-sm text-stone-500">Evaluation and policy before the engine's move</p>
          </div>
        </div>
        <div id="engine-analysis" class="space-y-3 text-sm text-stone-700">
          <p class="text-stone-500">No engine move yet.</p>
        </div>
      </section>

      <section class="rounded-xl border border-line bg-panel p-5 shadow-sm">
        <div class="mb-4 flex items-center justify-between gap-3">
          <div>
            <h2 class="text-xl font-bold">Current Position</h2>
            <p class="text-sm text-stone-500">Optional network-only probe</p>
          </div>
          <button id="analyze-current" class="rounded-lg border border-forest px-3 py-2 text-sm font-semibold text-forest hover:bg-paper">Analyze</button>
        </div>
        <div id="current-analysis" class="space-y-3 text-sm text-stone-700">
          <p class="text-stone-500">Press Analyze for the current board.</p>
        </div>
      </section>
    </aside>
  </section>
</main>
<script>{}</script>
</body>
</html>"#,
        escape_html(&title),
        escape_html(&title),
        escape_html(run_dir),
        play_js()
    )
}

pub fn render_bench_report(report: &BenchReport<'_>) -> String {
    let total = report.rows.len();
    let correct = report.rows.iter().filter(|r| r.correct).count();
    let accuracy = if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    };
    let rows = report
        .rows
        .iter()
        .enumerate()
        .map(|(i, row)| render_report_row(i, row))
        .collect::<String>();
    let categories = report
        .rows
        .iter()
        .filter_map(|row| row.category.as_deref())
        .fold(Vec::<&str>::new(), |mut acc, category| {
            if !acc.contains(&category) {
                acc.push(category);
            }
            acc
        })
        .into_iter()
        .map(|category| {
            format!(
                r#"<option value="{}">{}</option>"#,
                escape_attr(category),
                escape_html(category)
            )
        })
        .collect::<String>();

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{}</title>
<style>{}</style>
</head>
<body>
<main class="report">
  <header class="report-header">
    <div>
      <h1>{}</h1>
      <p class="muted">{} · model <code>{}</code> · {}</p>
    </div>
  </header>
  <section class="summary">
    <div><span>Total</span><strong>{}</strong></div>
    <div><span>Correct</span><strong>{}</strong></div>
    <div><span>Accuracy</span><strong>{:.1}%</strong></div>
  </section>
  <section class="toolbar">
    <input id="filter" placeholder="Filter by move, name, category">
    <select id="status-filter">
      <option value="all">All results</option>
      <option value="pass">Pass only</option>
      <option value="fail">Fail only</option>
    </select>
    <select id="category-filter">
      <option value="all">All categories</option>
      {}
    </select>
  </section>
  <section id="cases" class="cases">{}</section>
</main>
<script>{}</script>
</body>
</html>"#,
        escape_html(report.title),
        report_css(),
        escape_html(report.title),
        escape_html(game_name(report.game)),
        escape_html(report.model),
        escape_html(report.mode),
        total,
        correct,
        100.0 * accuracy,
        categories,
        rows,
        report_js()
    )
}

pub fn chess_board_for_position(position: &PositionSpec) -> anyhow::Result<String> {
    match position {
        PositionSpec::Chess(position) => Ok(ChessGame::from_position(position)?.to_string()),
        PositionSpec::Connect4(_) => anyhow::bail!("HTML board rendering is chess-only for now"),
    }
}

fn render_report_row(index: usize, row: &BenchReportRow) -> String {
    let title = row
        .name
        .clone()
        .unwrap_or_else(|| format!("Position {}", index + 1));
    let status = if row.correct { "pass" } else { "fail" };
    let expected = if row.expected.is_empty() {
        "none".to_owned()
    } else {
        row.expected.join(", ")
    };
    let category = row.category.as_deref().unwrap_or("uncategorized");
    let policy = row
        .analysis
        .policy
        .iter()
        .filter(|m| m.p > 0.0)
        .collect::<Vec<_>>()
        .into_iter()
        .take(12)
        .map(|m| {
            format!(
                r#"<div class="policy-row"><span>{}</span><div class="bar"><i style="width:{:.3}%"></i></div><strong>{:.1}%</strong></div>"#,
                escape_html(&m.mv),
                (100.0 * m.p).max(1.5),
                100.0 * m.p
            )
        })
        .collect::<String>();
    format!(
        r#"<article class="case" data-status="{status}" data-category="{}" data-text="{}">
  <button class="case-head" type="button">
    <span class="case-title"><strong>{}</strong><small>{}</small></span>
    <span class="best {}">{}</span>
    <span class="badge {status}">{}</span>
  </button>
  <div class="case-body">
    <div class="board-card">{}</div>
    <div class="facts">
      <div><span>Expected</span><strong>{}</strong></div>
      <div><span>Best</span><strong>{}</strong></div>
      <div><span>Value</span><strong>{:.3}</strong></div>
      <h3>Top policy</h3>
      {}
    </div>
  </div>
</article>"#,
        escape_attr(category),
        escape_attr(&format!(
            "{title} {category} {expected} {}",
            row.best_move.as_deref().unwrap_or("")
        )),
        escape_html(&title),
        escape_html(category),
        if row.correct { "ok" } else { "bad" },
        escape_html(row.best_move.as_deref().unwrap_or("none")),
        if row.correct { "pass" } else { "fail" },
        render_ascii_board(&row.board),
        escape_html(&expected),
        escape_html(row.best_move.as_deref().unwrap_or("none")),
        row.analysis.value,
        policy
    )
}

fn render_ascii_board(board: &str) -> String {
    let mut cells = String::new();
    for line in board.lines().take(8) {
        let mut parts = line.split_whitespace();
        let Some(rank) = parts.next() else {
            continue;
        };
        for (file, piece) in parts.take(8).enumerate() {
            let light = (rank.parse::<usize>().unwrap_or(1) + file).is_multiple_of(2);
            cells.push_str(&format!(
                r#"<div class="sq {}">{}</div>"#,
                if light { "light" } else { "dark" },
                piece_glyph(piece)
            ));
        }
    }
    format!(
        r#"<div class="chessboard">{cells}</div><pre>{}</pre>"#,
        escape_html(board)
    )
}

fn piece_glyph(piece: &str) -> &'static str {
    match piece {
        "P" => "♙",
        "N" => "♘",
        "B" => "♗",
        "R" => "♖",
        "Q" => "♕",
        "K" => "♔",
        "p" => "♟",
        "n" => "♞",
        "b" => "♝",
        "r" => "♜",
        "q" => "♛",
        "k" => "♚",
        _ => "",
    }
}

fn play_js() -> &'static str {
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

fn report_css() -> &'static str {
    r#":root{color-scheme:dark;--bg:#111315;--panel:#1b2024;--panel2:#22282d;--line:#343c43;--text:#f0f3f5;--muted:#9da7af;--ok:#57b98f;--bad:#e06a64;--accent:#d9a441;--bar:#395f50}*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#151819,#0f1112);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}.report{max-width:1240px;margin:0 auto;padding:28px}.report-header{margin-bottom:18px}h1{margin:0;font-size:32px;letter-spacing:0}.muted{color:var(--muted)}code{color:#d6dee5}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}.summary div,.case,.toolbar{background:var(--panel);border:1px solid var(--line);border-radius:8px}.summary div{padding:16px}.summary span,.facts span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase}.summary strong{font-size:30px}.toolbar{display:grid;grid-template-columns:2fr 180px 260px;gap:10px;padding:12px;margin-bottom:12px;position:sticky;top:0;z-index:2}input,select{width:100%;background:#252b30;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:9px 11px}.cases{display:grid;gap:10px}.case-head{width:100%;display:grid;grid-template-columns:1fr 110px auto;gap:12px;align-items:center;text-align:left;background:var(--panel);color:var(--text);border:0;padding:14px 16px;cursor:pointer}.case-title{display:grid;gap:3px}.case-title strong{font-size:16px}.case-title small{color:var(--accent)}.best{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:800}.best.ok{color:var(--ok)}.best.bad{color:var(--bad)}.badge{border-radius:999px;padding:4px 9px;font-weight:800;text-transform:uppercase;font-size:12px}.badge.pass{background:rgba(79,178,134,.16);color:var(--ok)}.badge.fail{background:rgba(217,93,93,.16);color:var(--bad)}.case-body{display:none;grid-template-columns:380px 1fr;gap:18px;padding:16px;border-top:1px solid var(--line);background:var(--panel2)}.case.open .case-body{display:grid}.chessboard{display:grid;grid-template-columns:repeat(8,1fr);aspect-ratio:1;border:1px solid var(--line);border-radius:6px;overflow:hidden}.sq{display:flex;align-items:center;justify-content:center;font-size:30px}.light{background:#d6c9a8;color:#111}.dark{background:#718568;color:#111}pre{white-space:pre-wrap;color:var(--muted);font-size:12px}.facts{display:grid;align-content:start;gap:12px}.facts strong{font-size:18px}.facts h3{margin:8px 0 0}.policy-row{display:grid;grid-template-columns:78px 1fr 58px;gap:10px;align-items:center}.policy-row span{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--text)}.bar{height:10px;background:#313941;border-radius:999px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--bar),var(--accent));border-radius:999px}@media(max-width:860px){.summary,.case-body,.toolbar{grid-template-columns:1fr}.case-head{grid-template-columns:1fr auto}.best{display:none}}"#
}

fn report_js() -> &'static str {
    r#"document.querySelectorAll('.case-head').forEach(b=>b.onclick=()=>b.closest('.case').classList.toggle('open'));const filter=document.getElementById('filter'),status=document.getElementById('status-filter'),category=document.getElementById('category-filter');function apply(){const q=filter.value.toLowerCase(),s=status.value,k=category.value;document.querySelectorAll('.case').forEach(c=>{const ok=(!q||c.dataset.text.toLowerCase().includes(q))&&(s==='all'||c.dataset.status===s)&&(k==='all'||c.dataset.category===k);c.style.display=ok?'':'none'})}filter.oninput=apply;status.onchange=apply;category.onchange=apply;"#
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn escape_attr(s: &str) -> String {
    escape_html(s).replace('\'', "&#39;")
}
