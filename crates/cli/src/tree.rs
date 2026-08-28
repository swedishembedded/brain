// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared tree renderer for `brain models {list,list-adapters,info}`: one
//! [`Node`] structure, two renderers built from the SAME data, so the piped
//! and interactive views can never show different information.
//!
//! - **Plain** ([`render_plain`]): box-drawing lines, the default whenever
//!   stdout is not a terminal (a pipe, a redirect, a file). Every LEAF node's
//!   [`Node::line`] is self-contained by construction (callers bake the full
//!   canonical id and every relevant column into it) - `grep`ping the output
//!   for a quant tag or a vendor/repo returns a complete, useful line on its
//!   own, never a fragment that only makes sense next to its parent. Header
//!   ("branch") nodes exist purely for the tree's visual structure and are
//!   not meant to be grepped for.
//! - **Interactive** ([`run_tui`]): a ratatui browser (arrow/`j`/`k`/
//!   `PageUp`/`PageDown` to move, `Enter`/`Space` to expand/collapse a branch
//!   OR open a leaf's detail view, `Esc` to go back a screen (or quit at the
//!   top), `/` to filter, `q`/`Ctrl-C` to quit), the default whenever stdout
//!   IS a terminal - mirrors `braintop`'s own split between its TUI and
//!   `--cli` paths, and its minimal-widget style (hand-built `Line`/`Span` +
//!   one `Paragraph`, no stateful `List`/`Table`).

use std::collections::HashSet;
use std::io::{self, IsTerminal};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;

/// One row. Leaf-vs-branch is implicit in `children.is_empty()` - see the
/// module doc for what that distinction means for each renderer.
pub struct Node {
    pub line: String,
    pub children: Vec<Node>,
    /// Set only on a leaf that [`run_tui`]'s `Enter` can "open" into a
    /// second screen - the canonical reference `on_enter` resolves. `None`
    /// for a branch (which always just expands/collapses) and for a leaf
    /// with nothing to open (a declared-but-not-pulled row - there is no
    /// checkpoint on disk to show tensors for).
    pub detail_ref: Option<String>,
}

impl Node {
    pub fn leaf(line: impl Into<String>) -> Node {
        Node { line: line.into(), children: Vec::new(), detail_ref: None }
    }
    pub fn branch(line: impl Into<String>, children: Vec<Node>) -> Node {
        Node { line: line.into(), children, detail_ref: None }
    }
    /// A leaf that `run_tui`'s `Enter` opens into a detail screen, resolved
    /// by calling the `on_enter` closure with `detail_ref`.
    pub fn leaf_with_detail(line: impl Into<String>, detail_ref: impl Into<String>) -> Node {
        Node { line: line.into(), children: Vec::new(), detail_ref: Some(detail_ref.into()) }
    }
}

pub fn stdout_is_terminal() -> bool {
    io::stdout().is_terminal()
}

// ----------------------------------------------------------------- plain --

pub fn render_plain(roots: &[Node]) -> Vec<String> {
    let mut out = Vec::new();
    for (i, n) in roots.iter().enumerate() {
        render_node(n, "", i + 1 == roots.len(), true, &mut out);
    }
    out
}

fn render_node(n: &Node, prefix: &str, last: bool, is_root: bool, out: &mut Vec<String>) {
    if is_root {
        out.push(n.line.clone());
    } else {
        out.push(format!("{prefix}{}{}", if last { "└─ " } else { "├─ " }, n.line));
    }
    let child_prefix = if is_root { String::new() } else { format!("{prefix}{}", if last { "   " } else { "│  " }) };
    for (i, c) in n.children.iter().enumerate() {
        render_node(c, &child_prefix, i + 1 == n.children.len(), false, out);
    }
}

// ------------------------------------------------------------- interactive --

/// A flattened row: the tree-connector prefix and the node's own path are
/// precomputed once (sibling structure never changes), so a redraw only has
/// to decide which rows are VISIBLE (collapsed ancestors / an active
/// filter), not re-walk the tree.
struct FlatRow {
    path: Vec<usize>,
    prefix: String,
    text: String,
    has_children: bool,
    detail_ref: Option<String>,
}

fn flatten(roots: &[Node]) -> Vec<FlatRow> {
    let mut out = Vec::new();
    for (i, n) in roots.iter().enumerate() {
        walk(n, vec![i], String::new(), i + 1 == roots.len(), true, &mut out);
    }
    out
}

fn walk(n: &Node, path: Vec<usize>, prefix: String, last: bool, is_root: bool, out: &mut Vec<FlatRow>) {
    let row_prefix = if is_root { String::new() } else { format!("{prefix}{}", if last { "└─ " } else { "├─ " }) };
    out.push(FlatRow { path: path.clone(), prefix: row_prefix, text: n.line.clone(), has_children: !n.children.is_empty(), detail_ref: n.detail_ref.clone() });
    let child_prefix = if is_root { String::new() } else { format!("{prefix}{}", if last { "   " } else { "│  " }) };
    for (i, c) in n.children.iter().enumerate() {
        let mut cp = path.clone();
        cp.push(i);
        walk(c, cp, child_prefix.clone(), i + 1 == n.children.len(), false, out);
    }
}

fn has_collapsed_ancestor(path: &[usize], collapsed: &HashSet<Vec<usize>>) -> bool {
    (1..path.len()).any(|k| collapsed.contains(&path[..k]))
}

/// Rows to show right now: honoring manual collapse/expand when no filter is
/// active, or - while filtering - every row that matches plus every row on
/// the path to a match (filtering deliberately ignores collapse state, so a
/// match under a collapsed branch is never hidden).
fn visible_rows<'a>(flat: &'a [FlatRow], collapsed: &HashSet<Vec<usize>>, filter: &str) -> Vec<&'a FlatRow> {
    if filter.is_empty() {
        return flat.iter().filter(|r| !has_collapsed_ancestor(&r.path, collapsed)).collect();
    }
    let needle = filter.to_lowercase();
    let matches: HashSet<usize> = flat.iter().enumerate().filter(|(_, r)| r.text.to_lowercase().contains(&needle)).map(|(i, _)| i).collect();
    flat.iter()
        .enumerate()
        .filter(|(i, r)| matches.contains(i) || flat.iter().enumerate().any(|(j, r2)| matches.contains(&j) && r2.path.starts_with(&r.path)))
        .map(|(_, r)| r)
        .collect()
}

/// `selected + delta`, clamped to `[0, len)` (or `0` when `len == 0`) - one
/// rule shared by `j`/`k` (`delta = ±1`) and `PageUp`/`PageDown`
/// (`delta = ∓area_height`), so a page move can never land past either end.
fn move_selection(selected: usize, delta: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    (selected as isize + delta).clamp(0, len as isize - 1) as usize
}

fn draw(f: &mut ratatui::Frame, visible: &[&FlatRow], selected: usize, scroll: usize, collapsed: &HashSet<Vec<usize>>, filter: &str, filtering: bool, can_go_back: bool) {
    let area = f.area();
    let lines: Vec<Line> = visible
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let marker = if r.has_children { if collapsed.contains(&r.path) { "▸ " } else { "▾ " } } else { "  " };
            let content = format!("{}{marker}{}", r.prefix, r.text);
            let style = if i == selected { Style::default().add_modifier(Modifier::REVERSED) } else { Style::default() };
            Line::from(Span::styled(content, style))
        })
        .collect();
    let title = if filtering {
        format!(" /{filter} ")
    } else {
        format!(
            " brain models - q quit · j/k/pgup/pgdn move · enter open/toggle · {} · / filter ",
            if can_go_back { "esc back" } else { "esc quit" }
        )
    };
    let p = Paragraph::new(lines).scroll((scroll as u16, 0)).block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(p, area);
}

/// Restore the terminal on panic so a crash never leaves the shell in
/// raw/alternate-screen mode - same fix `braintop` applies to its own TUI.
fn install_panic_hook() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        hook(info);
    }));
}

/// One browser screen's own navigation state - a stack of these is what
/// makes `Enter` on a leaf push a detail view and `Esc` pop back to exactly
/// where the list was left (selection, scroll, collapse and filter all
/// preserved), rather than losing your place.
struct Screen {
    flat: Vec<FlatRow>,
    collapsed: HashSet<Vec<usize>>,
    selected: usize,
    scroll: usize,
    filter: String,
    filtering: bool,
}

impl Screen {
    fn new(roots: &[Node]) -> Screen {
        Screen { flat: flatten(roots), collapsed: HashSet::new(), selected: 0, scroll: 0, filter: String::new(), filtering: false }
    }
}

/// Resolves a leaf's [`Node::detail_ref`] into a new screen's worth of
/// [`Node`]s - see [`run_tui`]'s doc.
pub type DetailFn = dyn Fn(&str) -> Vec<Node>;

/// The interactive browser. Blocks until the user quits (`q`/`Ctrl-C`, or
/// `Esc` with no screen left to go back to).
///
/// `on_enter`, if given, is called with a leaf's [`Node::detail_ref`] when
/// `Enter`/`Space` is pressed on a leaf that has one (a branch always just
/// expands/collapses, regardless of `on_enter`) - its returned tree becomes
/// a new screen, pushed onto a stack `Esc` pops. Pass `None` for a browser
/// with no drill-down (e.g. `list-adapters`, which has nothing further to
/// open on top of what it already shows).
pub fn run_tui(roots: Vec<Node>, on_enter: Option<&DetailFn>) -> io::Result<()> {
    let mut stack: Vec<Screen> = vec![Screen::new(&roots)];

    install_panic_hook();
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let result = (|| -> io::Result<()> {
        loop {
            let area_height = terminal.size()?.height.saturating_sub(2) as usize; // minus the border
            let screen = stack.last_mut().expect("at least one screen");
            let visible = visible_rows(&screen.flat, &screen.collapsed, &screen.filter);
            if visible.is_empty() {
                screen.selected = 0;
            } else if screen.selected >= visible.len() {
                screen.selected = visible.len() - 1;
            }
            if area_height > 0 {
                if screen.selected < screen.scroll {
                    screen.scroll = screen.selected;
                } else if screen.selected >= screen.scroll + area_height {
                    screen.scroll = screen.selected + 1 - area_height;
                }
            }
            let can_go_back = stack.len() > 1;

            terminal.draw(|f| {
                let screen = stack.last().expect("at least one screen");
                let visible = visible_rows(&screen.flat, &screen.collapsed, &screen.filter);
                draw(f, &visible, screen.selected, screen.scroll, &screen.collapsed, &screen.filter, screen.filtering, can_go_back);
            })?;

            if !event::poll(Duration::from_millis(200))? {
                continue;
            }
            let Event::Key(k) = event::read()? else { continue };
            if k.kind != KeyEventKind::Press {
                continue;
            }

            let screen = stack.last_mut().expect("at least one screen");
            if screen.filtering {
                match k.code {
                    KeyCode::Esc => {
                        screen.filtering = false;
                        screen.filter.clear();
                    }
                    KeyCode::Enter => screen.filtering = false,
                    KeyCode::Backspace => {
                        screen.filter.pop();
                    }
                    KeyCode::Char(c) => screen.filter.push(c),
                    _ => {}
                }
                continue;
            }

            let visible_len = visible_rows(&screen.flat, &screen.collapsed, &screen.filter).len();
            match k.code {
                KeyCode::Char('q') => break,
                KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => break,
                KeyCode::Esc => {
                    if stack.len() > 1 {
                        stack.pop();
                    } else {
                        break;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => screen.selected = move_selection(screen.selected, 1, visible_len),
                KeyCode::Up | KeyCode::Char('k') => screen.selected = move_selection(screen.selected, -1, visible_len),
                KeyCode::PageDown => screen.selected = move_selection(screen.selected, area_height.max(1) as isize, visible_len),
                KeyCode::PageUp => screen.selected = move_selection(screen.selected, -(area_height.max(1) as isize), visible_len),
                KeyCode::Enter | KeyCode::Char(' ') => {
                    let picked = visible_rows(&screen.flat, &screen.collapsed, &screen.filter)
                        .get(screen.selected)
                        .map(|r| (r.path.clone(), r.has_children, r.detail_ref.clone()));
                    if let Some((path, has_children, detail_ref)) = picked {
                        if has_children {
                            if !screen.collapsed.remove(&path) {
                                screen.collapsed.insert(path);
                            }
                        } else if let (Some(on_enter), Some(reference)) = (on_enter, detail_ref) {
                            let detail_nodes = on_enter(&reference);
                            stack.push(Screen::new(&detail_nodes));
                        }
                    }
                }
                KeyCode::Char('/') => {
                    screen.filtering = true;
                    screen.filter.clear();
                }
                _ => {}
            }
        }
        Ok(())
    })();

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Node> {
        vec![Node::branch(
            "qwen3",
            vec![Node::branch(
                "Qwen/Qwen3-0.6B",
                vec![Node::leaf("qwen3 Qwen/Qwen3-0.6B-Q8_0"), Node::leaf("qwen3 Qwen/Qwen3-0.6B-Q4_K_M")],
            )],
        )]
    }

    #[test]
    fn plain_render_every_leaf_line_is_self_contained_and_greppable() {
        let out = render_plain(&sample());
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], "qwen3");
        assert_eq!(out[1], "└─ Qwen/Qwen3-0.6B");
        // A grep for "Q4_K_M" must find a line that ALSO carries the arch and
        // repo, not just the bare quant token - the whole point of baking the
        // breadcrumb into the leaf's own `line`.
        let q4 = out.iter().find(|l| l.contains("Q4_K_M")).expect("Q4_K_M line present");
        assert!(q4.contains("qwen3") && q4.contains("Qwen/Qwen3-0.6B"), "leaf line {q4:?} is missing breadcrumb context");
    }

    #[test]
    fn plain_render_last_child_uses_the_corner_connector() {
        let out = render_plain(&sample());
        assert!(out[3].starts_with("   └─ "), "last leaf under a last branch should be double-indented with a corner: {:?}", out[3]);
        assert!(out[2].starts_with("   ├─ "), "first of two leaves should use a tee, not a corner: {:?}", out[2]);
    }

    #[test]
    fn flatten_preserves_one_row_per_node_with_correct_paths() {
        let flat = flatten(&sample());
        assert_eq!(flat.len(), 4);
        assert_eq!(flat[0].path, vec![0]);
        assert_eq!(flat[1].path, vec![0, 0]);
        assert_eq!(flat[2].path, vec![0, 0, 0]);
        assert_eq!(flat[3].path, vec![0, 0, 1]);
        assert!(flat[0].has_children && flat[1].has_children);
        assert!(!flat[2].has_children && !flat[3].has_children);
    }

    #[test]
    fn collapsing_a_branch_hides_its_descendants_only() {
        let flat = flatten(&sample());
        let mut collapsed = HashSet::new();
        collapsed.insert(vec![0, 0]); // the "Qwen/Qwen3-0.6B" branch
        let visible = visible_rows(&flat, &collapsed, "");
        assert_eq!(visible.len(), 2, "the root and the collapsed branch itself stay visible, its two leaves do not");
        assert_eq!(visible[0].text, "qwen3");
        assert_eq!(visible[1].text, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn filtering_ignores_collapse_and_shows_the_path_to_every_match() {
        let flat = flatten(&sample());
        let mut collapsed = HashSet::new();
        collapsed.insert(vec![0, 0]); // would normally hide both leaves
        let visible = visible_rows(&flat, &collapsed, "Q4_K_M");
        let texts: Vec<&str> = visible.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts, vec!["qwen3", "Qwen/Qwen3-0.6B", "qwen3 Qwen/Qwen3-0.6B-Q4_K_M"], "filtering must reveal the match's whole ancestor chain even though it sits under a collapsed branch, and must exclude the non-matching sibling leaf");
    }

    #[test]
    fn empty_filter_falls_back_to_normal_collapse_behavior() {
        let flat = flatten(&sample());
        let collapsed = HashSet::new();
        assert_eq!(visible_rows(&flat, &collapsed, "").len(), 4);
    }

    #[test]
    fn flatten_carries_detail_ref_only_on_the_nodes_that_set_it() {
        let roots = vec![Node::branch(
            "qwen3",
            vec![
                Node::leaf_with_detail("qwen3 Qwen/Qwen3-0.6B-Q8_0 local", "Qwen/Qwen3-0.6B-Q8_0"),
                Node::leaf("qwen3 Qwen/Qwen3-8B-Q8_0 not pulled"),
            ],
        )];
        let flat = flatten(&roots);
        assert_eq!(flat[0].detail_ref, None, "a branch never carries a detail_ref");
        assert_eq!(flat[1].detail_ref.as_deref(), Some("Qwen/Qwen3-0.6B-Q8_0"), "a pulled leaf's detail_ref must round-trip through flatten");
        assert_eq!(flat[2].detail_ref, None, "a not-pulled leaf has nothing to open");
    }

    #[test]
    fn move_selection_clamps_at_both_ends() {
        assert_eq!(move_selection(0, -1, 5), 0, "moving up from the top must stay at the top");
        assert_eq!(move_selection(4, 1, 5), 4, "moving down from the bottom must stay at the bottom");
        assert_eq!(move_selection(2, 1, 5), 3);
        assert_eq!(move_selection(2, -1, 5), 1);
    }

    #[test]
    fn move_selection_pages_by_the_given_delta_and_clamps() {
        // PageDown/PageUp pass ±area_height as delta - a page bigger than the
        // remaining distance must land exactly on the end, not overshoot past it.
        assert_eq!(move_selection(1, 20, 5), 4);
        assert_eq!(move_selection(3, -20, 5), 0);
        assert_eq!(move_selection(1, 2, 5), 3);
    }

    #[test]
    fn move_selection_on_an_empty_list_is_always_zero() {
        assert_eq!(move_selection(0, 1, 0), 0);
        assert_eq!(move_selection(0, -1, 0), 0);
    }
}
