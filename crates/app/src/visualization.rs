use crate::proxy::{game_name, GameKind};
use alphazero::Analysis;
use games::setup::GameSetup;
use games::ChessGame;

mod assets;

use assets::{play_js, report_css, report_js};

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
    pub position: GameSetup,
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
    }
    else {
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

pub fn chess_board_for_position(position: &GameSetup) -> anyhow::Result<String> {
    match position {
        GameSetup::Chess(position) => Ok(ChessGame::from_setup(position)?.to_string()),
        GameSetup::Connect4(_) => anyhow::bail!("HTML board rendering is chess-only for now"),
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
    }
    else {
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
        let Some(rank) = parts.next()
        else {
            continue;
        };
        for (file, piece) in parts.take(8).enumerate() {
            let light = (rank.parse::<usize>().unwrap_or(1) + file) % 2 == 0;
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

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn escape_attr(s: &str) -> String {
    escape_html(s).replace('\'', "&#39;")
}
