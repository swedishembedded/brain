// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Key-chord to action mapping. Longest chord wins: with W and W+Space both
//! mapped, holding W+Space picks the W+Space action. UX keys (pause, reset,
//! quality, quit) are consumed by the window layer BEFORE chord matching and
//! never appear in a chord.

/// Platform-neutral key identity used by the chord matcher. Only the keys a
/// world model can bind; UX keys live in [`UxKey`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    W,
    A,
    S,
    D,
    Space,
    Up,
    Down,
    Left,
    Right,
    /// Left shift — fly-camera sprint modifier.
    Shift,
    /// c — fly-camera move-down.
    C,
}

impl Key {
    /// Bit position inside a [`KeySet`] mask.
    fn bit(self) -> u16 {
        match self {
            Key::W => 1 << 0,
            Key::A => 1 << 1,
            Key::S => 1 << 2,
            Key::D => 1 << 3,
            Key::Space => 1 << 4,
            Key::Up => 1 << 5,
            Key::Down => 1 << 6,
            Key::Left => 1 << 7,
            Key::Right => 1 << 8,
            Key::Shift => 1 << 9,
            Key::C => 1 << 10,
        }
    }
}

/// A set of simultaneously pressed [`Key`]s (bitmask).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeySet(pub u16);

impl KeySet {
    pub fn empty() -> KeySet {
        KeySet(0)
    }
    pub fn of(keys: &[Key]) -> KeySet {
        KeySet(keys.iter().fold(0, |m, k| m | k.bit()))
    }
    pub fn press(&mut self, k: Key) {
        self.0 |= k.bit();
    }
    pub fn release(&mut self, k: Key) {
        self.0 &= !k.bit();
    }
    pub fn contains(self, other: KeySet) -> bool {
        self.0 & other.0 == other.0
    }
    pub fn len(self) -> u32 {
        self.0.count_ones()
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Chord table: pressed keys -> action id. Matching picks the mapped chord
/// with the MOST keys that is fully contained in the pressed set (ties break
/// to the earliest table entry); no contained chord -> `noop_action`.
#[derive(Clone, Debug)]
pub struct KeyChordMap {
    chords: Vec<(KeySet, u32)>,
    noop_action: u32,
}

impl KeyChordMap {
    pub fn new(chords: Vec<(KeySet, u32)>, noop_action: u32) -> KeyChordMap {
        KeyChordMap { chords, noop_action }
    }

    /// WASD -> actions 1..=4 (W,A,S,D), no chords — the FakeWorldModel map.
    pub fn wasd(noop_action: u32) -> KeyChordMap {
        KeyChordMap::new(
            vec![
                (KeySet::of(&[Key::W]), 1),
                (KeySet::of(&[Key::A]), 2),
                (KeySet::of(&[Key::S]), 3),
                (KeySet::of(&[Key::D]), 4),
            ],
            noop_action,
        )
    }

    /// Longest-chord-wins lookup for the current pressed set.
    pub fn action(&self, pressed: KeySet) -> u32 {
        let mut best: Option<(u32, u32)> = None; // (chord_len, action)
        for (chord, action) in &self.chords {
            if !chord.is_empty() && pressed.contains(*chord) {
                let len = chord.len();
                if best.is_none_or(|(bl, _)| len > bl) {
                    best = Some((len, *action));
                }
            }
        }
        best.map_or(self.noop_action, |(_, a)| a)
    }
}

/// UX keys handled by the window layer itself (never part of a chord).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UxKey {
    Quit,       // Esc / window close
    Reset,      // Enter
    Pause,      // .
    StepOnce,   // e (while paused)
    QualityUp,  // ]
    QualityDown, // [
    CycleView,  // v — cycle the demo's view mode (side / depth / stereo)
    Screenshot, // p — save the current frame
    ToggleMouse, // m — capture/release relative mouse-look
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keymap_single_keys_map_to_actions() {
        let m = KeyChordMap::wasd(0);
        let mut p = KeySet::empty();
        assert_eq!(m.action(p), 0);
        p.press(Key::W);
        assert_eq!(m.action(p), 1);
        p.release(Key::W);
        p.press(Key::D);
        assert_eq!(m.action(p), 4);
    }

    #[test]
    fn keymap_longest_chord_wins() {
        let m = KeyChordMap::new(
            vec![
                (KeySet::of(&[Key::W]), 1),
                (KeySet::of(&[Key::Space]), 5),
                (KeySet::of(&[Key::W, Key::Space]), 9),
            ],
            0,
        );
        assert_eq!(m.action(KeySet::of(&[Key::W])), 1);
        assert_eq!(m.action(KeySet::of(&[Key::W, Key::Space])), 9);
        // Extra unmapped keys in the pressed set do not break containment.
        assert_eq!(m.action(KeySet::of(&[Key::W, Key::Space, Key::A])), 9);
    }

    #[test]
    fn keymap_release_returns_to_noop() {
        let m = KeyChordMap::wasd(7);
        let mut p = KeySet::empty();
        p.press(Key::S);
        assert_eq!(m.action(p), 3);
        p.release(Key::S);
        assert_eq!(m.action(p), 7);
    }
}
