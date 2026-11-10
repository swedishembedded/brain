// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decoder's line grammar.
//!
//! After an optional `<think>...</think>` block the decoder writes one line
//! per item, in section order:
//!
//! ```text
//! S key=value; key=value                      scene
//! E<id> <name> @<position> *<slot> {k=v; ...} an entity (with *slot: has a box)
//! T *<slot>                                   a text line
//! R <subject id> <predicate> <object id>      a relation
//! V <text>                                    an event
//! ```
//!
//! Lines must arrive with non-decreasing section rank and unique keys; the
//! first line that breaks either ends the output. [`rank`] and [`key`] are
//! shared by generation (to stop early) and parsing (to drop what follows).

/// Section rank of a line by its first character (`-1` for anything else).
/// An entity line with a `*` slot is an object (rank 1), without one a
/// region (rank 2).
pub fn rank(line: &str) -> i32 {
    match line.chars().next() {
        Some('S') => 0,
        Some('E') => {
            if line.contains(" *") {
                1
            } else {
                2
            }
        }
        Some('T') => 3,
        Some('R') => 4,
        Some('V') => 5,
        _ => -1,
    }
}

/// Identity of a line for duplicate detection: scene lines are all one item,
/// entities are identified by their id, every other line by its whole text.
pub fn key(line: &str) -> String {
    match line.chars().next() {
        Some('S') => "S".to_string(),
        Some('E') => line.split(' ').next().unwrap_or(line).to_string(),
        _ => line.to_string(),
    }
}

/// An entity line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entity {
    /// The `E` index.
    pub id: u32,
    /// What it is.
    pub name: String,
    /// Where it is, if stated.
    pub position: Option<String>,
    /// The proposal slot it references; `None` for a region.
    pub slot: Option<usize>,
    /// Attributes in order.
    pub description: Vec<(String, String)>,
}

/// A relation line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relation {
    /// Subject entity id.
    pub subject: u32,
    /// The predicate words.
    pub predicate: String,
    /// Object entity id.
    pub object: u32,
}

/// A decoded text split into its items.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Parsed {
    /// The text inside `<think>...</think>`, when the block was opened.
    pub reasoning: Option<String>,
    /// Whether the think block was closed.
    pub think_closed: bool,
    /// Scene attributes in order.
    pub scene: Vec<(String, String)>,
    /// Entities in order.
    pub entities: Vec<Entity>,
    /// Text proposal slots in order.
    pub texts: Vec<usize>,
    /// Relations in order.
    pub relations: Vec<Relation>,
    /// Event lines (text after `V `).
    pub events: Vec<String>,
}

fn pairs(text: &str) -> Vec<(String, String)> {
    text.split(';')
        .filter_map(|kv| {
            let kv = kv.trim();
            kv.split_once('=').map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        })
        .collect()
}

fn entity(line: &str) -> Option<Entity> {
    let rest = line.strip_prefix('E')?;
    let (id, rest) = rest.split_once(' ')?;
    let id: u32 = id.parse().ok()?;
    let (body, description) = match rest.find(" {") {
        Some(i) => (&rest[..i], pairs(rest[i + 2..].trim_end().strip_suffix('}')?)),
        None => (rest, Vec::new()),
    };
    let (body, slot) = match body.rfind(" *") {
        Some(i) => match body[i + 2..].trim().parse::<usize>() {
            Ok(k) => (&body[..i], Some(k)),
            Err(_) => (body, None),
        },
        None => (body, None),
    };
    let (name, position) = match body.find(" @") {
        Some(i) => (&body[..i], Some(body[i + 2..].to_string())),
        None => (body, None),
    };
    (!name.is_empty()).then(|| Entity { id, name: name.to_string(), position, slot, description })
}

/// Split decoder output into items, applying the line-order rules.
pub fn parse(text: &str) -> Parsed {
    let mut parsed = Parsed::default();
    let mut rest = text;
    if let Some(after) = text.strip_prefix("<think>") {
        match after.split_once("</think>") {
            Some((think, tail)) => {
                parsed.reasoning = Some(think.to_string());
                parsed.think_closed = true;
                rest = tail;
            }
            None => {
                parsed.reasoning = Some(after.to_string());
                return parsed;
            }
        }
    } else {
        parsed.think_closed = true;
    }
    let (mut last_rank, mut seen) = (i32::MIN, std::collections::BTreeSet::new());
    for line in rest.split('\n').filter(|l| !l.is_empty()) {
        let (r, k) = (rank(line), key(line));
        if r < 0 || r < last_rank || !seen.insert(k) {
            break;
        }
        last_rank = r;
        match line.chars().next() {
            Some('S') => parsed.scene = pairs(line[1..].trim_start()),
            Some('E') => parsed.entities.extend(entity(line)),
            Some('T') => parsed.texts.extend(line.trim_start_matches('T').trim().strip_prefix('*').and_then(|k| k.trim().parse::<usize>().ok())),
            Some('R') => {
                let words: Vec<&str> = line[1..].split_whitespace().collect();
                if let (Some(first), Some(last)) = (words.first(), words.last()) {
                    if words.len() >= 3 {
                        if let (Ok(subject), Ok(object)) = (first.parse(), last.parse()) {
                            parsed.relations.push(Relation { subject, predicate: words[1..words.len() - 1].join(" "), object });
                        }
                    }
                }
            }
            Some('V') => parsed.events.push(line[1..].trim().to_string()),
            _ => {}
        }
    }
    parsed
}
