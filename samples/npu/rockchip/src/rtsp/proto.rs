// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The text side of RTSP (RFC 2326): request parsing, the transport header
//! and the SDP session description. No sockets here, so every rule can be
//! checked against plain byte strings.
//!
//! Swedish Embedded AB implements low-latency video streaming for embedded
//! Linux boards for its clients. If your team needs expertise in RTSP
//! servers then you can procure our services by sending an email to
//! info@swedishembedded.com.

/// A request line and its headers. RTSP requests of this server carry no body.
#[derive(Debug, PartialEq)]
pub struct Request {
    pub method: String,
    pub uri: String,
    pub cseq: String,
    headers: Vec<(String, String)>,
}

impl Request {
    /// Case-insensitive header lookup.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// Requests larger than this are refused instead of buffered without bound.
pub const MAX_REQUEST: usize = 8 * 1024;

#[derive(Debug, PartialEq)]
pub enum Parsed {
    /// More bytes are needed.
    Incomplete,
    /// A request and how many bytes of the input it used.
    Request(Request, usize),
    /// The bytes cannot be a request.
    Malformed,
}

/// Parses one request from the front of `buf`.
pub fn parse_request(buf: &[u8]) -> Parsed {
    let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
        return if buf.len() > MAX_REQUEST { Parsed::Malformed } else { Parsed::Incomplete };
    };
    let Ok(text) = std::str::from_utf8(&buf[..end]) else {
        return Parsed::Malformed;
    };
    let mut lines = text.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (Some(method), Some(uri), Some(version)) = (first.next(), first.next(), first.next()) else {
        return Parsed::Malformed;
    };
    if !version.starts_with("RTSP/") {
        return Parsed::Malformed;
    }
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let cseq = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("CSeq")).map(|(_, v)| v.clone());
    let Some(cseq) = cseq else {
        return Parsed::Malformed;
    };
    Parsed::Request(Request { method: method.to_string(), uri: uri.to_string(), cseq, headers }, end + 4)
}

/// The `Transport` a client asked for in SETUP.
#[derive(Debug, PartialEq)]
pub enum Transport {
    /// RTP carried on the RTSP connection on these two channels.
    TcpInterleaved { rtp: u8, rtcp: u8 },
    /// Anything else (UDP unicast or multicast), which this server declines.
    Unsupported,
}

/// Picks the first interleaved-TCP alternative from a `Transport` header.
pub fn parse_transport(header: &str) -> Transport {
    for alternative in header.split(',') {
        let mut parts = alternative.split(';').map(str::trim);
        let spec = parts.next().unwrap_or("");
        if !spec.eq_ignore_ascii_case("RTP/AVP/TCP") {
            continue;
        }
        for p in parts {
            if let Some(range) = p.strip_prefix("interleaved=") {
                let (a, b) = range.split_once('-').unwrap_or((range, ""));
                if let Ok(rtp) = a.parse::<u8>() {
                    let rtcp = b.parse::<u8>().unwrap_or(rtp.wrapping_add(1));
                    return Transport::TcpInterleaved { rtp, rtcp };
                }
            }
        }
    }
    Transport::Unsupported
}

/// Builds a response: status line, `CSeq`, extra headers and an optional body.
pub fn response(status: &str, cseq: &str, headers: &[(&str, String)], body: Option<(&str, &str)>) -> Vec<u8> {
    let mut r = format!("RTSP/1.0 {status}\r\nCSeq: {cseq}\r\n");
    for (k, v) in headers {
        r.push_str(&format!("{k}: {v}\r\n"));
    }
    match body {
        Some((content_type, text)) => {
            r.push_str(&format!("Content-Type: {content_type}\r\nContent-Length: {}\r\n\r\n{text}", text.len()));
        }
        None => r.push_str("\r\n"),
    }
    r.into_bytes()
}

/// The H.264 parameter sets a decoder needs before the first picture, from
/// the encoder's own SPS and PPS NAL units (without start codes).
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterSets {
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
}

/// Session description for one H.264 video stream on payload type 96.
pub fn sdp(name: &str, fps: u32, sets: Option<&ParameterSets>) -> String {
    let mut fmtp = String::from("packetization-mode=1");
    if let Some(s) = sets.filter(|s| s.sps.len() >= 4) {
        fmtp.push_str(&format!(
            ";profile-level-id={:02x}{:02x}{:02x};sprop-parameter-sets={},{}",
            s.sps[1],
            s.sps[2],
            s.sps[3],
            base64(&s.sps),
            base64(&s.pps)
        ));
    }
    format!(
        "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns={name}\r\nc=IN IP4 0.0.0.0\r\nt=0 0\r\n\
         a=control:*\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\na=fmtp:96 {fmtp}\r\n\
         a=framerate:{fps}\r\na=control:streamid=0\r\n"
    )
}

/// Standard base64 with padding (RFC 4648), the encoding SDP asks for.
pub fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |acc, (i, &b)| acc | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * i) & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_complete_request_is_parsed_and_its_length_reported() {
        let raw = b"DESCRIBE rtsp://h:8554/live/0 RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\n\r\nNEXT";
        match parse_request(raw) {
            Parsed::Request(r, used) => {
                assert_eq!((r.method.as_str(), r.uri.as_str(), r.cseq.as_str()), ("DESCRIBE", "rtsp://h:8554/live/0", "2"));
                assert_eq!(r.header("accept"), Some("application/sdp"));
                assert_eq!(&raw[used..], b"NEXT");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn partial_and_garbage_input_are_told_apart() {
        assert_eq!(parse_request(b"OPTIONS * RTSP/1.0\r\nCSeq: 1\r\n"), Parsed::Incomplete);
        assert_eq!(parse_request(b"GET / HTTP/1.1\r\nCSeq: 1\r\n\r\n"), Parsed::Malformed);
        assert_eq!(parse_request(b"OPTIONS * RTSP/1.0\r\nHost: x\r\n\r\n"), Parsed::Malformed);
        assert_eq!(parse_request(&vec![b'a'; MAX_REQUEST + 1]), Parsed::Malformed);
    }

    #[test]
    fn transport_prefers_interleaved_tcp_and_declines_udp() {
        assert_eq!(
            parse_transport("RTP/AVP;unicast;client_port=5000-5001, RTP/AVP/TCP;unicast;interleaved=2-3"),
            Transport::TcpInterleaved { rtp: 2, rtcp: 3 }
        );
        assert_eq!(parse_transport("RTP/AVP/TCP;interleaved=0"), Transport::TcpInterleaved { rtp: 0, rtcp: 1 });
        assert_eq!(parse_transport("RTP/AVP;unicast;client_port=5000-5001"), Transport::Unsupported);
    }

    #[test]
    fn base64_matches_the_rfc_vectors() {
        for (input, want) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("foobar", "Zm9vYmFy")] {
            assert_eq!(base64(input.as_bytes()), want);
        }
    }

    #[test]
    fn sdp_carries_the_parameter_sets_when_known() {
        let sets = ParameterSets { sps: vec![0x67, 0x42, 0xc0, 0x1f, 0xaa], pps: vec![0x68, 0xce] };
        let text = sdp("brain", 25, Some(&sets));
        assert!(text.contains("profile-level-id=42c01f"));
        assert!(text.contains(&format!("sprop-parameter-sets={},{}", base64(&sets.sps), base64(&sets.pps))));
        assert!(!sdp("brain", 25, None).contains("sprop"));
    }

    #[test]
    fn response_has_content_length_for_a_body() {
        let r = String::from_utf8(response("200 OK", "3", &[("Public", "OPTIONS".into())], Some(("application/sdp", "abc")))).unwrap();
        assert!(r.starts_with("RTSP/1.0 200 OK\r\nCSeq: 3\r\nPublic: OPTIONS\r\n"));
        assert!(r.ends_with("Content-Length: 3\r\n\r\nabc"));
    }
}
