// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The game's framebuffer, as it comes off the socket.
//!
//! 320x200 palette indices and the 256-entry RGB palette in effect when they
//! were drawn, both base64 in a JSON object. Indexed rather than RGB because
//! it is a fifth of the bytes; the palette travels with them because the
//! engine tints its own output - the red wash when the player is hit is a
//! palette change and not a pixel change, so a fixed table would show a calm
//! picture at exactly the moment worth looking at.
//!
//! The frame is for HUMANS. Nothing the model reads comes from here.

use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    width: u32,
    height: u32,
    format: String,
    pixels: String,
    palette: String,
}

#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// One palette index per pixel, row-major.
    pub pixels: Vec<u8>,
    /// 256 RGB triples.
    pub palette: Vec<u8>,
}

impl Frame {
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    pub fn parse(json: &str) -> Result<Frame, String> {
        let w: Wire = serde_json::from_str(json).map_err(|e| format!("unreadable frame: {e}"))?;
        if w.format != "indexed8" {
            return Err(format!("the game sent a {:?} frame, which this cannot draw", w.format));
        }
        let pixels = base64(&w.pixels)?;
        let palette = base64(&w.palette)?;
        let want = (w.width * w.height) as usize;
        if pixels.len() != want {
            return Err(format!("frame says {}x{} but carries {} bytes", w.width, w.height, pixels.len()));
        }
        if palette.len() < 768 {
            return Err(format!("a 256-colour palette needs 768 bytes, got {}", palette.len()));
        }
        Ok(Frame { width: w.width, height: w.height, pixels, palette })
    }
}

/// Decode standard base64. Rejects anything that is not, rather than skipping
/// it: a frame that silently decodes half its pixels is a frame that looks
/// like a rendering bug.
fn base64(s: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for (i, b) in s.bytes().enumerate() {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\n' | b'\r' => continue,
            other => return Err(format!("byte {i} of the frame is {other:?}, not base64")),
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_encoder_on_every_tail_length() {
        // The three residues are where a hand-rolled codec goes wrong, and the
        // symptom is a frame that is right except for its last pixels - which
        // reads as a rendering artefact rather than as a decoder bug.
        for (encoded, expect) in
            [("", &b""[..]), ("QQ==", b"A"), ("QUI=", b"AB"), ("QUJD", b"ABC"), ("QUJDRA==", b"ABCD")]
        {
            assert_eq!(base64(encoded).expect("decodes"), expect, "{encoded:?}");
        }
        assert!(base64("QQ*=").is_err(), "a non-base64 byte must be an error");
    }

    #[test]
    fn a_truncated_frame_is_rejected_not_drawn() {
        let json = r#"{"width":2,"height":2,"format":"indexed8","pixels":"QQ==",
                       "palette":"QQ=="}"#;
        let e = Frame::parse(json).expect_err("must not accept 1 byte for 4 pixels");
        assert!(e.contains("carries 1 bytes"), "{e}");
    }
}
