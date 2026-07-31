// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rendering — the progressive-reveal dashboard, the device-colored residency
//! bars, and the drill-in detail views. Pure `&App -> Frame`; no state mutation.
//!
//! Everything is data-driven from the [`StatsSnapshot`]: N accelerators render N
//! rows, N models render N bars, and any `extra` map renders as a generic
//! key→value tree, so metrics added upstream appear with no change here.

use std::collections::BTreeMap;

use brain_stats::{Accelerator, ModelStat, StatsSnapshot};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use serde_json::Value;

use crate::app::{App, Focus, View};

/// Residency colors, per the spec: CPU = red, NPU = yellow, GPU = green.
pub fn device_color(device: &str) -> Color {
    let d = device.to_ascii_lowercase();
    if d.starts_with("cpu") {
        Color::Red
    } else if d.starts_with("npu") {
        Color::Yellow
    } else if d.starts_with("gpu") {
        Color::Green
    } else {
        Color::Gray
    }
}

/// Human-readable byte size (binary units).
pub fn fmt_bytes(b: u64) -> String {
    const K: f64 = 1024.0;
    let f = b as f64;
    if f >= K * K * K {
        format!("{:.1}G", f / (K * K * K))
    } else if f >= K * K {
        format!("{:.0}M", f / (K * K))
    } else if f >= K {
        format!("{:.0}K", f / K)
    } else {
        format!("{b}B")
    }
}

/// Top-level entry point: draw the whole frame for the current app state.
pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let [header, body, footer] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(3), Constraint::Length(1)]).areas(area);

    draw_header(f, header, app);
    draw_footer(f, footer, app);

    match &app.snapshot {
        None => draw_waiting(f, body, app),
        Some(snap) => match app.view {
            View::Dashboard => draw_dashboard(f, body, app, snap),
            View::Detail => draw_detail(f, body, app, snap),
        },
    }
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let dot = if app.connected { Span::styled("●", Style::default().fg(Color::Green)) } else { Span::styled("●", Style::default().fg(Color::Red)) };
    let line = Line::from(vec![
        Span::styled("braintop", Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan)),
        Span::raw("  "),
        dot,
        Span::raw(" "),
        Span::styled(app.name.clone(), Style::default().fg(Color::Gray)),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let keys = match app.view {
        View::Dashboard => "q quit  Tab panel  j/k move  Enter drill-in",
        View::Detail => "q quit  Esc/h back",
    };
    let line = Line::from(vec![
        Span::styled(keys, Style::default().fg(Color::DarkGray)),
        Span::raw("   "),
        Span::styled(app.status.clone(), Style::default().fg(if app.connected { Color::DarkGray } else { Color::Yellow })),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn draw_waiting(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::ALL).title(" braintop ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("waiting for brain ({})…", app.name),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "start it with:  brain serve --dbus",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(""),
        Line::from(Span::styled(app.status.clone(), Style::default().fg(Color::DarkGray))),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

// ---- dashboard -------------------------------------------------------------

fn draw_dashboard(f: &mut Frame, area: Rect, app: &App, snap: &StatsSnapshot) {
    // Responsive vertical stack: fixed-ish chrome panels + Models grows to fill.
    let acc_h = (snap.accelerators.len() as u16 + 2).clamp(3, 12);
    let exec_h = 4 + snap.extra.len().min(4) as u16;
    let req_h = (snap.requests.len().max(1) as u16 + 2).clamp(3, 8);
    let [acc_a, model_a, exec_a, req_a, conn_a] = Layout::vertical([
        Constraint::Length(acc_h),
        Constraint::Min(4),
        Constraint::Length(exec_h),
        Constraint::Length(req_h),
        Constraint::Length(3),
    ])
    .areas(area);

    draw_accelerators(f, acc_a, app, snap);
    draw_models(f, model_a, app, snap);
    draw_executor(f, exec_a, snap);
    draw_requests(f, req_a, app, snap);
    draw_connections(f, conn_a, app, snap);
}

/// A memory bar with a reserved overlay, `width` cells wide.
fn mem_bar_spans(width: usize, used: u64, reserved: u64, total: u64) -> Vec<Span<'static>> {
    if total == 0 || width == 0 {
        return vec![Span::styled("·".repeat(width), Style::default().fg(Color::DarkGray))];
    }
    let u = ((used as f64 / total as f64) * width as f64).round() as usize;
    let u = u.min(width);
    let r = ((reserved as f64 / total as f64) * width as f64).round() as usize;
    let r = r.min(width - u);
    let free = width - u - r;
    vec![
        Span::styled("█".repeat(u), Style::default().fg(Color::Cyan)),
        Span::styled("▒".repeat(r), Style::default().fg(Color::Magenta)),
        Span::styled("·".repeat(free), Style::default().fg(Color::DarkGray)),
    ]
}

fn accelerator_line(a: &Accelerator, selected: bool, width: u16) -> Line<'static> {
    let marker = if selected { "▶ " } else { "  " };
    // Reserve space for label + numbers, give the rest to the bar.
    let barw = (width as usize).saturating_sub(40).clamp(6, 40);
    let mut spans = vec![
        Span::styled(marker, Style::default().fg(Color::Cyan)),
        Span::styled(format!("{:<6}", a.id), Style::default().fg(device_color(&a.id)).add_modifier(Modifier::BOLD)),
        Span::raw(" ["),
    ];
    spans.extend(mem_bar_spans(barw, a.mem_used, a.mem_reserved, a.mem_total));
    spans.push(Span::raw("] "));
    spans.push(Span::styled(
        format!("{}/{}", fmt_bytes(a.mem_used), fmt_bytes(a.mem_total)),
        Style::default().fg(Color::White),
    ));
    if let Some(u) = a.util {
        spans.push(Span::styled(format!("  {u:>4.0}%"), Style::default().fg(Color::LightGreen)));
    }
    let mut line = Line::from(spans);
    if selected {
        line = line.style(Style::default().add_modifier(Modifier::REVERSED));
    }
    line
}

fn draw_accelerators(f: &mut Frame, area: Rect, app: &App, snap: &StatsSnapshot) {
    let block = panel_block(" Accelerators ", app.focus == Focus::Accelerators);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let sel = app.sel(Focus::Accelerators);
    let focused = app.focus == Focus::Accelerators;
    let lines: Vec<Line> = snap
        .accelerators
        .iter()
        .enumerate()
        .map(|(i, a)| accelerator_line(a, focused && i == sel, inner.width))
        .collect();
    f.render_widget(Paragraph::new(lines), inner);
}

/// One model's residency bar: a colored run per device instance (CPU red / NPU
/// yellow / GPU green), widths ∝ per-instance memory, scaled by `max_total` so
/// bars are comparable. Non-resident models render dimmed.
fn model_line(m: &ModelStat, selected: bool, max_total: u64, width: u16) -> Line<'static> {
    let marker = if selected { "▶ " } else { "  " };
    let barw = (width as usize).saturating_sub(34).clamp(6, 40);
    let mut spans = vec![
        Span::styled(marker, Style::default().fg(Color::Cyan)),
        Span::styled(format!("{:<14}", truncate(&m.id, 14)), Style::default().add_modifier(Modifier::BOLD)),
    ];
    if !m.resident || m.instances.is_empty() {
        spans.push(Span::styled(format!("{:<width$}", "cold", width = barw + 2), Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM)));
    } else {
        spans.push(Span::raw("["));
        let total: u64 = m.instances.iter().map(|i| i.mem).sum();
        let scale = if max_total > 0 { max_total } else { total.max(1) };
        for inst in &m.instances {
            let w = ((inst.mem as f64 / scale as f64) * barw as f64).round() as usize;
            let w = w.max(1);
            spans.push(Span::styled("█".repeat(w), Style::default().fg(device_color(&inst.device))));
        }
        spans.push(Span::raw("] "));
        // A compact per-device legend (device:size), colored.
        for inst in &m.instances {
            spans.push(Span::styled(
                format!("{}:{} ", inst.device, fmt_bytes(inst.mem)),
                Style::default().fg(device_color(&inst.device)),
            ));
        }
    }
    spans.push(Span::styled(format!(" {}", m.family), Style::default().fg(Color::DarkGray)));
    let mut line = Line::from(spans);
    if selected {
        line = line.style(Style::default().add_modifier(Modifier::REVERSED));
    }
    line
}

fn draw_models(f: &mut Frame, area: Rect, app: &App, snap: &StatsSnapshot) {
    let block = panel_block(" Models / residency ", app.focus == Focus::Models);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let sel = app.sel(Focus::Models);
    let focused = app.focus == Focus::Models;
    let max_total = snap
        .models
        .iter()
        .map(|m| m.instances.iter().map(|i| i.mem).sum::<u64>())
        .max()
        .unwrap_or(0);
    let lines: Vec<Line> = if snap.models.is_empty() {
        vec![Line::from(Span::styled("no models in catalog", Style::default().fg(Color::DarkGray)))]
    } else {
        snap.models
            .iter()
            .enumerate()
            .map(|(i, m)| model_line(m, focused && i == sel, max_total, inner.width))
            .collect()
    };
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_executor(f: &mut Frame, area: Rect, snap: &StatsSnapshot) {
    let block = Block::default().borders(Borders::ALL).title(" Executor ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    let e = &snap.executor;
    let counter = |k: &str, v: u64| Span::styled(format!("{k} {v}  "), Style::default().fg(Color::White));
    let mut lines = vec![Line::from(vec![
        counter("builds", e.builds),
        counter("evict", e.evictions),
        counter("batches", e.batches),
        counter("jobs", e.jobs),
        counter("resident", e.resident),
        counter("queue⤒", e.queue_peak),
        counter("max_batch", e.max_batch),
        counter("max_par", e.max_parallel),
    ])];
    // Snapshot-level `extra` renders here as a generic tree (compact on dashboard).
    lines.extend(extra_lines("", &snap.extra));
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_requests(f: &mut Frame, area: Rect, app: &App, snap: &StatsSnapshot) {
    let block = panel_block(" Requests in progress ", app.focus == Focus::Requests);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let sel = app.sel(Focus::Requests);
    let focused = app.focus == Focus::Requests;
    let lines: Vec<Line> = if snap.requests.is_empty() {
        vec![Line::from(Span::styled("no active requests", Style::default().fg(Color::DarkGray)))]
    } else {
        snap.requests
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let marker = if focused && i == sel { "▶ " } else { "  " };
                let who = r.model.clone().or_else(|| r.provider.clone()).unwrap_or_else(|| r.id.clone());
                let spin = spinner(r.since_ms);
                let mut line = Line::from(vec![
                    Span::styled(marker, Style::default().fg(Color::Cyan)),
                    Span::styled(format!("{spin} "), Style::default().fg(Color::LightMagenta)),
                    Span::styled(format!("{:<12}", truncate(&r.id, 12)), Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(format!("{:<10}", r.phase), phase_style(&r.phase)),
                    Span::styled(format!("{:<16}", truncate(&who, 16)), Style::default().fg(Color::White)),
                    Span::styled(format!("{}ms", r.since_ms), Style::default().fg(Color::DarkGray)),
                ]);
                if focused && i == sel {
                    line = line.style(Style::default().add_modifier(Modifier::REVERSED));
                }
                line
            })
            .collect()
    };
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_connections(f: &mut Frame, area: Rect, app: &App, snap: &StatsSnapshot) {
    let block = panel_block(" Connections ", app.focus == Focus::Connections);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let line = if snap.connections.is_empty() {
        Line::from(Span::styled("no connections", Style::default().fg(Color::DarkGray)))
    } else {
        Line::from(Span::styled(format!("{} connection(s)", snap.connections.len()), Style::default().fg(Color::White)))
    };
    f.render_widget(Paragraph::new(line), inner);
}

// ---- detail (drill-in) -----------------------------------------------------

fn draw_detail(f: &mut Frame, area: Rect, app: &App, snap: &StatsSnapshot) {
    let (title, lines) = match app.focus {
        Focus::Models => model_detail(snap, app.sel(Focus::Models)),
        Focus::Accelerators => accelerator_detail(snap, app.sel(Focus::Accelerators)),
        Focus::Requests => request_detail(snap, app.sel(Focus::Requests)),
        Focus::Connections => connection_detail(snap, app.sel(Focus::Connections)),
    };
    let block = Block::default().borders(Borders::ALL).title(title).border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    f.render_widget(block, area);
    f.render_widget(Paragraph::new(lines), inner);
}

fn model_detail(snap: &StatsSnapshot, i: usize) -> (String, Vec<Line<'static>>) {
    let Some(m) = snap.models.get(i) else {
        return (" model ".into(), vec![Line::from("no model")]);
    };
    let mut lines = vec![
        kv("id", &m.id),
        kv("family", &m.family),
        kv("resident", &m.resident.to_string()),
        kv("capabilities", &m.capabilities.join(", ")),
        Line::from(""),
        Line::from(Span::styled("instances (device / tier / mem):", Style::default().add_modifier(Modifier::BOLD))),
    ];
    if m.instances.is_empty() {
        lines.push(Line::from(Span::styled("  (cold — no resident instances)", Style::default().fg(Color::DarkGray))));
    }
    for inst in &m.instances {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{:<7}", inst.device), Style::default().fg(device_color(&inst.device)).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{:<6}", inst.tier), Style::default().fg(Color::White)),
            Span::styled(fmt_bytes(inst.mem), Style::default().fg(Color::White)),
        ]));
        lines.extend(indent(extra_lines("", &inst.extra), 4));
    }
    lines.extend(extra_section(&m.extra));
    (format!(" model: {} ", m.id), lines)
}

fn accelerator_detail(snap: &StatsSnapshot, i: usize) -> (String, Vec<Line<'static>>) {
    let Some(a) = snap.accelerators.get(i) else {
        return (" accelerator ".into(), vec![Line::from("no accelerator")]);
    };
    let mut lines = vec![
        kv("id", &a.id),
        kv("kind", &a.kind),
        kv("name", &a.name),
        kv("index", &a.index.to_string()),
        kv("mem_total", &fmt_bytes(a.mem_total)),
        kv("mem_used", &fmt_bytes(a.mem_used)),
        kv("mem_reserved", &fmt_bytes(a.mem_reserved)),
    ];
    if let Some(u) = a.util {
        lines.push(kv("util", &format!("{u:.1}%")));
    }
    lines.extend(extra_section(&a.extra));
    (format!(" accelerator: {} ", a.id), lines)
}

fn request_detail(snap: &StatsSnapshot, i: usize) -> (String, Vec<Line<'static>>) {
    let Some(r) = snap.requests.get(i) else {
        return (" request ".into(), vec![Line::from(Span::styled("no active requests", Style::default().fg(Color::DarkGray)))]);
    };
    let mut lines = vec![
        kv("id", &r.id),
        kv("provider", r.provider.as_deref().unwrap_or("-")),
        kv("model", r.model.as_deref().unwrap_or("-")),
        kv("action", r.action.as_deref().unwrap_or("-")),
        kv("phase", &r.phase),
        kv("since_ms", &r.since_ms.to_string()),
    ];
    lines.extend(extra_section(&r.extra));
    (format!(" request: {} ", r.id), lines)
}

fn connection_detail(snap: &StatsSnapshot, i: usize) -> (String, Vec<Line<'static>>) {
    let Some(c) = snap.connections.get(i) else {
        return (" connections ".into(), vec![Line::from(Span::styled("no connections", Style::default().fg(Color::DarkGray)))]);
    };
    let mut lines = vec![kv("id", &c.id)];
    lines.extend(extra_section(&c.extra));
    (format!(" connection: {} ", c.id), lines)
}

// ---- shared helpers --------------------------------------------------------

fn panel_block(title: &'static str, focused: bool) -> Block<'static> {
    let mut b = Block::default().borders(Borders::ALL).title(title);
    if focused {
        b = b.border_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    }
    b
}

fn kv(k: &str, v: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{k:<14}"), Style::default().fg(Color::DarkGray)),
        Span::styled(v.to_string(), Style::default().fg(Color::White)),
    ])
}

fn extra_section(extra: &BTreeMap<String, Value>) -> Vec<Line<'static>> {
    if extra.is_empty() {
        return vec![];
    }
    let mut lines = vec![Line::from(""), Line::from(Span::styled("extra:", Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow)))];
    lines.extend(indent(extra_lines("", extra), 2));
    lines
}

/// Render an `extra` map as a generic key→value tree — nested objects indent,
/// arrays index, scalars are leaves. New metrics show up here automatically.
pub fn extra_lines(prefix: &str, extra: &BTreeMap<String, Value>) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for (k, v) in extra {
        value_lines(&join(prefix, k), v, &mut out);
    }
    out
}

fn value_lines(key: &str, v: &Value, out: &mut Vec<Line<'static>>) {
    match v {
        Value::Object(map) => {
            for (k, val) in map {
                value_lines(&join(key, k), val, out);
            }
        }
        Value::Array(items) => {
            for (i, val) in items.iter().enumerate() {
                value_lines(&join(key, &i.to_string()), val, out);
            }
        }
        scalar => {
            let sval = match scalar {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            out.push(Line::from(vec![
                Span::styled(format!("{key} "), Style::default().fg(Color::Yellow)),
                Span::styled(sval, Style::default().fg(Color::White)),
            ]));
        }
    }
}

fn join(prefix: &str, k: &str) -> String {
    if prefix.is_empty() {
        k.to_string()
    } else {
        format!("{prefix}.{k}")
    }
}

fn indent(lines: Vec<Line<'static>>, n: usize) -> Vec<Line<'static>> {
    let pad = " ".repeat(n);
    lines
        .into_iter()
        .map(|mut l| {
            l.spans.insert(0, Span::raw(pad.clone()));
            l
        })
        .collect()
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}

fn phase_style(phase: &str) -> Style {
    let c = match phase {
        "running" => Color::Green,
        "queued" => Color::Yellow,
        _ => Color::Gray,
    };
    Style::default().fg(c)
}

/// A tiny time-driven spinner so long-running requests visibly animate.
fn spinner(since_ms: u64) -> char {
    const FRAMES: [char; 4] = ['⠋', '⠙', '⠹', '⠸'];
    FRAMES[((since_ms / 120) % FRAMES.len() as u64) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests_support::sample_snapshot;
    use crate::app::App;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn render_to_string(app: &App, w: u16, h: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    #[test]
    fn dashboard_renders_ids_names_and_extra_without_panic() {
        let app = App::preview(sample_snapshot());
        let s = render_to_string(&app, 160, 50);
        // Model ids.
        assert!(s.contains("qwen"), "expected model id 'qwen' in buffer");
        assert!(s.contains("tts"), "expected model id 'tts' in buffer");
        // Accelerator ids/names.
        assert!(s.contains("gpu0"), "expected accelerator id 'gpu0' in buffer");
        assert!(s.contains("cpu"), "expected accelerator id 'cpu' in buffer");
        // A snapshot-level `extra` key rendered generically on the dashboard.
        assert!(s.contains("uptime_ms"), "expected extra key 'uptime_ms' in buffer");
    }

    #[test]
    fn waiting_state_renders_without_a_snapshot() {
        let app = App::new("com.swedishembedded.Brain1");
        let s = render_to_string(&app, 100, 30);
        assert!(s.contains("waiting for brain"));
    }

    #[test]
    fn model_detail_shows_per_device_instances_and_extra() {
        let mut app = App::preview(sample_snapshot());
        app.handle_key(crossterm::event::KeyCode::Enter, false); // drill into model 0 (qwen)
        let s = render_to_string(&app, 120, 30);
        assert!(s.contains("model: qwen"));
        assert!(s.contains("hot"));
        assert!(s.contains("warm"));
        assert!(s.contains("params")); // model-level extra key
    }

    #[test]
    fn tiny_terminal_does_not_panic() {
        // Layout must survive a cramped terminal (constraints, not fixed math).
        let app = App::preview(sample_snapshot());
        let _ = render_to_string(&app, 20, 8);
    }
}
