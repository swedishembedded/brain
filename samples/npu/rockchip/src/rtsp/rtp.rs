// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RTP packetization of H.264 access units (RFC 6184, packetization mode 1).
//!
//! An access unit in Annex B form (NAL units separated by start codes) becomes
//! a run of RTP packets: a NAL unit that fits goes out as a single-NAL packet,
//! a larger one is split into FU-A fragments. Every packet is written already
//! wrapped in the four-byte `$ channel length` header that RTSP uses to carry
//! RTP over the control TCP connection (RFC 2326 section 10.12), because that
//! is the transport the server speaks.
//!
//! Swedish Embedded AB implements low-latency video streaming for embedded
//! Linux boards for its clients. If your team needs expertise in RTP and
//! RTSP then you can procure our services by sending an email to
//! info@swedishembedded.com.

/// Largest RTP payload; with the 12-byte RTP header and the 4-byte
/// interleaved header a packet stays under a 1500-byte Ethernet MTU with room
/// for TCP and IP headers.
pub const MAX_PAYLOAD: usize = 1400;
const INTERLEAVE_HEADER: usize = 4;
const NAL_FU_A: u8 = 28;
const NAL_AUD: u8 = 9;

/// Splits an Annex B access unit into its NAL units, without start codes and
/// without trailing zero bytes (which belong to the next start code).
pub fn split_nals(au: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= au.len() {
        if au[i] == 0 && au[i + 1] == 0 && au[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::with_capacity(starts.len());
    for (n, &start) in starts.iter().enumerate() {
        // The next NAL's start code begins three bytes before its payload.
        let mut end = starts.get(n + 1).map_or(au.len(), |next| next - 3);
        while end > start && au[end - 1] == 0 {
            end -= 1;
        }
        if end > start {
            nals.push(&au[start..end]);
        }
    }
    nals
}

/// The packets of one access unit, back to back in one buffer.
#[derive(Default)]
pub struct Packets {
    buf: Vec<u8>,
    /// Offset of each packet's interleaved header within `buf`.
    headers: Vec<usize>,
}

impl Packets {
    pub fn clear(&mut self) {
        self.buf.clear();
        self.headers.clear();
    }

    /// All packets, ready to be written to a socket in one call.
    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Rewrites the interleaved channel of every packet, for a client that
    /// asked for a channel other than the one the packets were built with.
    pub fn set_channel(&mut self, channel: u8) {
        for &h in &self.headers {
            self.buf[h + 1] = channel;
        }
    }

    fn begin(&mut self, marker: bool, payload_type: u8, seq: u16, timestamp: u32, ssrc: u32) {
        self.headers.push(self.buf.len());
        self.buf.extend_from_slice(&[b'$', 0, 0, 0]); // length patched in `end`
        self.buf.push(0x80); // V=2, no padding, no extension, no CSRC
        self.buf.push(u8::from(marker) << 7 | payload_type);
        self.buf.extend_from_slice(&seq.to_be_bytes());
        self.buf.extend_from_slice(&timestamp.to_be_bytes());
        self.buf.extend_from_slice(&ssrc.to_be_bytes());
    }

    fn end(&mut self) {
        let h = *self.headers.last().expect("end follows begin");
        let rtp_len = (self.buf.len() - h - INTERLEAVE_HEADER) as u16;
        self.buf[h + 2..h + 4].copy_from_slice(&rtp_len.to_be_bytes());
    }
}

/// Turns access units into RTP packets of one stream.
pub struct Packetizer {
    payload_type: u8,
    ssrc: u32,
    seq: u16,
}

impl Packetizer {
    pub fn new(payload_type: u8, ssrc: u32) -> Self {
        Packetizer { payload_type, ssrc, seq: 0 }
    }

    /// Appends the packets for `au` to `out`.
    /// `timestamp` is on the 90 kHz video clock. The last packet carries the
    /// RTP marker bit that ends the access unit.
    pub fn packetize(&mut self, au: &[u8], timestamp: u32, out: &mut Packets) {
        let nals: Vec<&[u8]> = split_nals(au).into_iter().filter(|n| n[0] & 0x1f != NAL_AUD).collect();
        for (n, nal) in nals.iter().enumerate() {
            let last_nal = n + 1 == nals.len();
            if nal.len() <= MAX_PAYLOAD {
                self.emit(out, last_nal, timestamp, &[], nal);
            } else {
                let (header, body) = (nal[0], &nal[1..]);
                let chunk = MAX_PAYLOAD - 2;
                let pieces = body.chunks(chunk).count();
                for (p, piece) in body.chunks(chunk).enumerate() {
                    let start = p == 0;
                    let end = p + 1 == pieces;
                    let fu = [
                        (header & 0xe0) | NAL_FU_A,
                        u8::from(start) << 7 | u8::from(end) << 6 | (header & 0x1f),
                    ];
                    self.emit(out, last_nal && end, timestamp, &fu, piece);
                }
            }
        }
    }

    fn emit(&mut self, out: &mut Packets, marker: bool, timestamp: u32, prefix: &[u8], payload: &[u8]) {
        out.begin(marker, self.payload_type, self.seq, timestamp, self.ssrc);
        out.buf.extend_from_slice(prefix);
        out.buf.extend_from_slice(payload);
        out.end();
        self.seq = self.seq.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nal(kind: u8, len: usize) -> Vec<u8> {
        let mut v = vec![0x60 | kind];
        v.extend((0..len - 1).map(|i| (i % 251) as u8 + 1)); // never zero
        v
    }

    fn annex_b(nals: &[Vec<u8>]) -> Vec<u8> {
        let mut au = Vec::new();
        for (i, n) in nals.iter().enumerate() {
            au.extend_from_slice(if i == 0 { &[0, 0, 0, 1][..] } else { &[0, 0, 1][..] });
            au.extend_from_slice(n);
        }
        au
    }

    /// Splits `Packets` back into (marker, seq, timestamp, payload) tuples.
    fn parse(p: &Packets) -> Vec<(bool, u16, u32, Vec<u8>)> {
        let mut out = Vec::new();
        let b = p.bytes();
        let mut i = 0;
        while i < b.len() {
            assert_eq!(b[i], b'$');
            let len = u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
            let rtp = &b[i + 4..i + 4 + len];
            assert_eq!(rtp[0], 0x80);
            out.push((
                rtp[1] & 0x80 != 0,
                u16::from_be_bytes([rtp[2], rtp[3]]),
                u32::from_be_bytes([rtp[4], rtp[5], rtp[6], rtp[7]]),
                rtp[12..].to_vec(),
            ));
            i += 4 + len;
        }
        out
    }

    #[test]
    fn split_handles_three_and_four_byte_start_codes() {
        let a = nal(7, 10);
        let b = nal(8, 5);
        let c = nal(5, 30);
        let au = annex_b(&[a.clone(), b.clone(), c.clone()]);
        assert_eq!(split_nals(&au), vec![&a[..], &b[..], &c[..]]);
    }

    #[test]
    fn small_nals_go_out_whole_with_the_marker_on_the_last() {
        let (a, b) = (nal(7, 12), nal(5, 100));
        let mut pk = Packetizer::new(96, 0xdead_beef);
        let mut out = Packets::default();
        pk.packetize(&annex_b(&[a.clone(), b.clone()]), 9000, &mut out);
        let got = parse(&out);
        assert_eq!(got.len(), 2);
        assert_eq!((got[0].0, got[1].0), (false, true));
        assert_eq!((got[0].1, got[1].1), (0, 1));
        assert!(got.iter().all(|g| g.2 == 9000));
        assert_eq!((&got[0].3, &got[1].3), (&a, &b));
    }

    #[test]
    fn a_large_nal_is_fragmented_and_reassembles_exactly() {
        let big = nal(5, 5000);
        let mut pk = Packetizer::new(96, 1);
        let mut out = Packets::default();
        pk.packetize(&annex_b(&[big.clone()]), 0, &mut out);
        let got = parse(&out);
        assert!(got.len() > 3);
        let mut rebuilt = Vec::new();
        for (i, (marker, _, _, payload)) in got.iter().enumerate() {
            assert!(payload.len() <= MAX_PAYLOAD);
            assert_eq!(payload[0] & 0x1f, NAL_FU_A);
            assert_eq!(payload[1] & 0x80 != 0, i == 0, "start flag");
            assert_eq!(payload[1] & 0x40 != 0, i + 1 == got.len(), "end flag");
            assert_eq!(*marker, i + 1 == got.len());
            if i == 0 {
                rebuilt.push((payload[0] & 0xe0) | (payload[1] & 0x1f));
            }
            rebuilt.extend_from_slice(&payload[2..]);
        }
        assert_eq!(rebuilt, big);
    }

    #[test]
    fn the_channel_can_be_rewritten_for_every_packet() {
        let mut pk = Packetizer::new(96, 1);
        let mut out = Packets::default();
        pk.packetize(&annex_b(&[nal(5, 4000)]), 0, &mut out);
        out.set_channel(6);
        let b = out.bytes();
        let mut i = 0;
        while i < b.len() {
            assert_eq!(b[i + 1], 6);
            i += 4 + u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
        }
    }

    #[test]
    fn access_unit_delimiters_are_not_sent() {
        let mut pk = Packetizer::new(96, 1);
        let mut out = Packets::default();
        pk.packetize(&annex_b(&[nal(9, 2), nal(1, 20)]), 0, &mut out);
        let got = parse(&out);
        assert_eq!(got.len(), 1);
        assert!(got[0].0, "the only packet ends the access unit");
    }
}
