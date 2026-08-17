// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pure UI state — the latest [`StatsSnapshot`] plus the current view/selection.
//!
//! No I/O lives here: the [`client`](crate::client) task feeds [`Update`]s in via
//! [`App::apply`], and [`ui`](crate::ui) reads the state out. That keeps the whole
//! app testable against an in-memory snapshot with no D-Bus.

use brain_stats::StatsSnapshot;
use crossterm::event::KeyCode;

use crate::client::Update;

/// Which panel currently has the keyboard selection. `Tab`/`BackTab` cycles this
/// set in order; `Enter` drills into the selected row of the focused panel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Focus {
    Models,
    Accelerators,
    Requests,
    Connections,
}

impl Focus {
    /// Cycle order (also the per-panel selection-index order).
    pub const ALL: [Focus; 4] = [Focus::Models, Focus::Accelerators, Focus::Requests, Focus::Connections];

    fn idx(self) -> usize {
        Focus::ALL.iter().position(|f| *f == self).unwrap_or(0)
    }
    fn next(self) -> Focus {
        Focus::ALL[(self.idx() + 1) % Focus::ALL.len()]
    }
    fn prev(self) -> Focus {
        Focus::ALL[(self.idx() + Focus::ALL.len() - 1) % Focus::ALL.len()]
    }
}

/// Top-level view: the dashboard, or a drill-in detail of the focused selection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    Dashboard,
    Detail,
}

/// The whole braintop UI state.
pub struct App {
    /// Bus name we are talking to (for the header / waiting message).
    pub name: String,
    /// True once the D-Bus proxy is live.
    pub connected: bool,
    /// Last transport status line (error / "connected" / "waiting…").
    pub status: String,
    /// The most recent snapshot, if any has arrived.
    pub snapshot: Option<StatsSnapshot>,
    /// Which panel is focused.
    pub focus: Focus,
    /// Dashboard vs. a drill-in detail.
    pub view: View,
    /// Selected row per panel, indexed by [`Focus::ALL`] order.
    pub selected: [usize; 4],
    /// Set when the user asks to quit.
    pub should_quit: bool,
}

impl App {
    /// A fresh app, not yet connected, showing the waiting state.
    pub fn new(name: impl Into<String>) -> App {
        let name = name.into();
        App {
            status: format!("waiting for brain ({name})…"),
            name,
            connected: false,
            snapshot: None,
            focus: Focus::Models,
            view: View::Dashboard,
            selected: [0; 4],
            should_quit: false,
        }
    }

    /// A ready-to-render app seeded with a snapshot (used by tests / previews).
    pub fn preview(snapshot: StatsSnapshot) -> App {
        let mut app = App::new("com.swedishembedded.Brain1");
        app.connected = true;
        app.status = "connected".into();
        app.snapshot = Some(snapshot);
        app
    }

    /// Apply one transport update from the client task.
    pub fn apply(&mut self, u: Update) {
        match u {
            Update::Connected => {
                self.connected = true;
                self.status = "connected".into();
            }
            Update::Disconnected(why) => {
                self.connected = false;
                self.status = format!("waiting for brain ({}): {why}", self.name);
            }
            Update::Snapshot(s) => {
                self.connected = true;
                self.status = "connected".into();
                self.snapshot = Some(*s);
                self.clamp_selection();
            }
        }
    }

    /// Number of selectable rows in the focused panel.
    pub fn focus_len(&self, focus: Focus) -> usize {
        let Some(s) = &self.snapshot else { return 0 };
        match focus {
            Focus::Models => s.models.len(),
            Focus::Accelerators => s.accelerators.len(),
            Focus::Requests => s.requests.len(),
            Focus::Connections => s.connections.len(),
        }
    }

    /// The selected index within a panel (clamped for safety at read time).
    pub fn sel(&self, focus: Focus) -> usize {
        let len = self.focus_len(focus);
        if len == 0 {
            0
        } else {
            self.selected[focus.idx()].min(len - 1)
        }
    }

    fn clamp_selection(&mut self) {
        for f in Focus::ALL {
            let len = self.focus_len(f);
            let i = &mut self.selected[f.idx()];
            *i = if len == 0 { 0 } else { (*i).min(len - 1) };
        }
    }

    fn move_sel(&mut self, delta: i32) {
        let len = self.focus_len(self.focus);
        if len == 0 {
            return;
        }
        let cur = self.selected[self.focus.idx()].min(len - 1) as i32;
        let next = (cur + delta).clamp(0, len as i32 - 1) as usize;
        self.selected[self.focus.idx()] = next;
    }

    /// Handle one key press. `ctrl` is whether Ctrl was held.
    pub fn handle_key(&mut self, code: KeyCode, ctrl: bool) {
        match code {
            KeyCode::Char('c') if ctrl => self.should_quit = true,
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Up | KeyCode::Char('k') => self.move_sel(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_sel(1),
            KeyCode::Tab => self.focus = self.focus.next(),
            KeyCode::BackTab => self.focus = self.focus.prev(),
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if self.focus_len(self.focus) > 0 {
                    self.view = View::Detail;
                }
            }
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => self.view = View::Dashboard,
            _ => {}
        }
    }
}

/// Test fixtures shared by the `cli`/`ui` unit tests (built with `cfg(test)` in
/// mind but exposed so sibling test modules reuse one snapshot).
#[cfg(test)]
pub mod tests_support {
    use brain_stats::{Accelerator, ExecutorStat, Instance, ModelStat, RequestStat, StatsSnapshot};
    use serde_json::json;

    const GB: u64 = 1 << 30;

    /// A fixed snapshot: 2 accelerators (cpu + gpu0), 2 models (one resident across
    /// CPU+GPU, one cold), executor counters, and `extra` entries at two levels.
    pub fn sample_snapshot() -> StatsSnapshot {
        let mut snap = StatsSnapshot::new();
        snap.extra.insert("uptime_ms".into(), json!(1234));

        let mut cpu = Accelerator {
            id: "cpu".into(),
            kind: "cpu".into(),
            name: "CPU".into(),
            index: 0,
            mem_total: 16 * GB,
            mem_used: 4 * GB,
            mem_reserved: GB,
            util: Some(12.0),
            ..Default::default()
        };
        cpu.extra.insert("threads".into(), json!(8));

        let mut gpu = Accelerator {
            id: "gpu0".into(),
            kind: "gpu".into(),
            name: "GPU 0".into(),
            index: 0,
            mem_total: 24 * GB,
            mem_used: 8 * GB,
            mem_reserved: 2 * GB,
            util: Some(73.5),
            ..Default::default()
        };
        gpu.extra.insert("temp_c".into(), json!(61));
        snap.accelerators.push(cpu);
        snap.accelerators.push(gpu);

        // Resident across CPU (warm) + GPU (hot).
        let mut qwen = ModelStat {
            id: "qwen".into(),
            family: "qwen3".into(),
            capabilities: vec!["chat".into(), "embed".into()],
            resident: true,
            instances: vec![
                Instance { device: "gpu0".into(), tier: "hot".into(), mem: 6 * GB, ..Default::default() },
                Instance { device: "cpu".into(), tier: "warm".into(), mem: 2 * GB, ..Default::default() },
            ],
            ..Default::default()
        };
        qwen.extra.insert("params".into(), json!("7B"));
        snap.models.push(qwen);

        // Cold model — non-resident, dimmed in the UI.
        snap.models.push(ModelStat {
            id: "tts".into(),
            family: "tts".into(),
            capabilities: vec!["synth".into()],
            resident: false,
            instances: vec![],
            ..Default::default()
        });

        snap.executor = ExecutorStat {
            builds: 3,
            evictions: 1,
            batches: 12,
            jobs: 40,
            resident: 2,
            queue_peak: 5,
            max_batch: 4,
            max_parallel: 2,
            ..Default::default()
        };

        // One in-flight request so the requests panel/detail has data to render.
        snap.requests.push(RequestStat {
            id: "req1".into(),
            provider: Some("openai".into()),
            model: Some("qwen".into()),
            action: Some("chat".into()),
            phase: "running".into(),
            since_ms: 350,
            ..Default::default()
        });

        snap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tests_support::sample_snapshot;

    #[test]
    fn navigation_moves_within_focused_panel_and_cycles_focus() {
        let mut app = App::preview(sample_snapshot());
        assert_eq!(app.focus, Focus::Models);
        // Two models — down moves to index 1 and clamps there.
        app.handle_key(KeyCode::Down, false);
        assert_eq!(app.sel(Focus::Models), 1);
        app.handle_key(KeyCode::Down, false);
        assert_eq!(app.sel(Focus::Models), 1);
        app.handle_key(KeyCode::Up, false);
        assert_eq!(app.sel(Focus::Models), 0);
        // Tab cycles to the next panel.
        app.handle_key(KeyCode::Tab, false);
        assert_eq!(app.focus, Focus::Accelerators);
    }

    #[test]
    fn enter_drills_in_and_esc_returns() {
        let mut app = App::preview(sample_snapshot());
        app.handle_key(KeyCode::Enter, false);
        assert_eq!(app.view, View::Detail);
        app.handle_key(KeyCode::Esc, false);
        assert_eq!(app.view, View::Dashboard);
    }

    #[test]
    fn ctrl_c_and_q_quit() {
        let mut app = App::preview(sample_snapshot());
        app.handle_key(KeyCode::Char('c'), true);
        assert!(app.should_quit);
        let mut app2 = App::preview(sample_snapshot());
        app2.handle_key(KeyCode::Char('q'), false);
        assert!(app2.should_quit);
    }

    #[test]
    fn empty_panel_enter_is_a_noop() {
        // No snapshot → nothing selectable → Enter must not switch to Detail.
        let mut app = App::new("com.swedishembedded.Brain1");
        app.handle_key(KeyCode::Enter, false);
        assert_eq!(app.view, View::Dashboard);
    }
}
