// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end proof that a real forecast produces **identical `events::Event`
//! frames** whether driven over the in-process stdio pump or over a Unix socket.
//! This is the guarantee that makes the transport swappable under a client.

use fcbench::RandomWalk;
use forecast::{ForecastSpec, Panel, Representation, Variate};
use server::{pump_connection, serve_unix, ControllerSession, ServeOpts, Session};
use std::io::{BufRead, BufReader, Cursor, Write};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::thread;

fn naive_controller() -> runtime::Controller {
    let mut reg = runtime::Registry::new();
    reg.register_forecast(Arc::new(RandomWalk));
    runtime::Controller::new(reg)
}

/// One forecast_request line with a fixed req_id and deterministic payload.
fn request_line() -> String {
    let panel =
        Panel::single("1d", "AAPL", vec![Variate::target("close", vec![10.0, 11.0, 12.0, 13.0])]);
    let spec = ForecastSpec {
        horizon: 4,
        representations: vec![Representation::Quantiles, Representation::Point],
        quantile_levels: vec![0.1, 0.5, 0.9],
        num_samples: 0,
        seed: 7,
    };
    let env = events::Envelope::with_id(
        Some("r1".into()),
        events::Event::ForecastRequest { model: "naive".into(), panel, spec },
    );
    events::encode_envelope(&env)
}

/// Drive the request through the in-memory stdio pump; return the non-greeting
/// response lines.
fn over_stdio(req: &str) -> Vec<String> {
    let mut session = ControllerSession::new(naive_controller());
    let input = format!("{req}\n");
    let mut out = Vec::new();
    pump_connection(Cursor::new(input.into_bytes()), &mut out, &mut session).unwrap();
    String::from_utf8(out)
        .unwrap()
        .lines()
        .filter(|l| !l.contains("\"ready\""))
        .map(|s| s.to_string())
        .collect()
}

/// Drive the same request over a real Unix socket; return the non-greeting
/// response lines.
fn over_socket(req: &str) -> Vec<String> {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("brain-parity-{}.sock", std::process::id()));
    let p2 = path.clone();
    let make: server::transport::SessionFactory =
        Arc::new(|| Box::new(ControllerSession::new(naive_controller())));
    thread::spawn(move || {
        let _ = serve_unix(&p2, make, ServeOpts::default());
    });

    // connect (retry until the listener is up)
    let mut client = None;
    for _ in 0..200 {
        if let Ok(s) = UnixStream::connect(&path) {
            client = Some(s);
            break;
        }
        thread::sleep(std::time::Duration::from_millis(5));
    }
    let stream = client.expect("server never came up");
    let mut writer = stream.try_clone().unwrap();
    let mut reader = BufReader::new(stream);

    // consume greeting
    let mut ready = String::new();
    reader.read_line(&mut ready).unwrap();

    writeln!(writer, "{req}").unwrap();
    writer.flush().unwrap();

    // read exactly one response line (forecast is one-shot)
    let mut resp = String::new();
    reader.read_line(&mut resp).unwrap();
    let _ = std::fs::remove_file(&path);
    vec![resp.trim_end().to_string()]
}

#[test]
fn forecast_frames_are_identical_over_stdio_and_socket() {
    let req = request_line();
    let stdio = over_stdio(&req);
    let socket = over_socket(&req);
    assert_eq!(stdio.len(), 1, "expected one forecast_result over stdio: {stdio:?}");
    assert_eq!(socket.len(), 1, "expected one forecast_result over socket: {socket:?}");
    assert_eq!(stdio[0], socket[0], "the same request must yield byte-identical frames");
    // and it's actually a forecast_result carrying the req_id
    assert!(stdio[0].contains("\"forecast_result\""));
    assert!(stdio[0].contains("\"req_id\":\"r1\""));
}
