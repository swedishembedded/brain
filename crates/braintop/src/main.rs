// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `braintop` — the binary entry point.
//!
//! Parses args, then either runs the interactive TUI (default) or the `--cli`
//! one-shot flat dump. All rendering/state logic lives in the `braintop` library
//! crate; this file is just arg parsing + the terminal lifecycle + the event loop.

use std::io::{self, Stdout, Write};
use std::time::Duration;

use anyhow::Result;
use braintop::app::App;
use braintop::client::{self, Bus, ConnOpts};
use braintop::{cli, ui};
use crossterm::event::{Event, EventStream, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

const HELP: &str = "\
braintop — a btop-like live monitor for brain's serving state (over D-Bus).

USAGE:
    braintop [OPTIONS]

OPTIONS:
    --cli               Print one flattened `path=value` snapshot to stdout and exit.
    --system            Use the system bus (default: session bus).
    --address <ADDR>    Connect to an explicit D-Bus address instead of a well-known bus.
    --name <NAME>       Well-known name to talk to (default: com.swedishembedded.Brain1).
    --path <PATH>       Object path (default: /com/swedishembedded/Brain1).
    -h, --help          Show this help.

KEYS (TUI):
    q / Ctrl-C   quit          Tab   cycle panels     j/k or ↑/↓  move selection
    Enter        drill in      Esc/h back
";

struct Args {
    cli: bool,
    help: bool,
    opts: ConnOpts,
}

fn parse_args() -> Result<Args> {
    let mut cli = false;
    let mut help = false;
    let mut opts = ConnOpts::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--cli" => cli = true,
            "--system" => opts.bus = Bus::System,
            "--address" => {
                let v = it.next().ok_or_else(|| anyhow::anyhow!("--address needs a value"))?;
                opts.bus = Bus::Address(v);
            }
            "--name" => opts.name = it.next().ok_or_else(|| anyhow::anyhow!("--name needs a value"))?,
            "--path" => opts.path = it.next().ok_or_else(|| anyhow::anyhow!("--path needs a value"))?,
            "-h" | "--help" => help = true,
            other => anyhow::bail!("unknown argument: {other} (try --help)"),
        }
    }
    Ok(Args { cli, help, opts })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    if args.help {
        print!("{HELP}");
        return Ok(());
    }
    if args.cli {
        return run_cli(&args.opts).await;
    }
    run_tui(args.opts).await
}

/// `--cli`: fetch one snapshot and print flattened `path=value` lines.
async fn run_cli(opts: &ConnOpts) -> Result<()> {
    let snap = client::fetch_once(opts).await?;
    let mut out = io::stdout().lock();
    for line in cli::flatten_snapshot(&snap) {
        writeln!(out, "{line}")?;
    }
    Ok(())
}

/// The interactive TUI: set up the terminal, spawn the client task, run the event
/// loop, and always restore the terminal on the way out.
async fn run_tui(opts: ConnOpts) -> Result<()> {
    install_panic_hook();
    let mut terminal = setup_terminal()?;
    let result = event_loop(&mut terminal, opts).await;
    restore_terminal(&mut terminal)?;
    result
}

async fn event_loop(terminal: &mut Terminal<CrosstermBackend<Stdout>>, opts: ConnOpts) -> Result<()> {
    let name = opts.name.clone();
    let mut app = App::new(name);

    // Background D-Bus client → updates over an unbounded channel.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(client::run(opts, tx));

    let mut events = EventStream::new();
    // A steady redraw tick keeps spinners/animation live even between updates.
    let mut render = tokio::time::interval(Duration::from_millis(200));

    loop {
        terminal.draw(|f| ui::draw(f, &app))?;
        if app.should_quit {
            return Ok(());
        }
        tokio::select! {
            maybe_ev = events.next() => {
                match maybe_ev {
                    Some(Ok(Event::Key(key))) if key.kind != KeyEventKind::Release => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        app.handle_key(key.code, ctrl);
                    }
                    Some(Ok(Event::Resize(_, _))) => {}
                    Some(Err(_)) | None => return Ok(()),
                    _ => {}
                }
            }
            update = rx.recv() => {
                // `None` means the client task ended; keep showing the last state.
                if let Some(u) = update {
                    app.apply(u);
                }
            }
            _ = render.tick() => {}
        }
    }
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

/// Restore the terminal on panic so a crash never leaves it in raw/alt mode.
fn install_panic_hook() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        hook(info);
    }));
}
