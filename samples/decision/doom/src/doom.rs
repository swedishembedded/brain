// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The game, as a subprocess and a socket.
//!
//! RESTful-DOOM is the 1993 engine with an HTTP API inside its game loop. This
//! module owns its whole lifetime: find the binary, pick a free port, start it
//! headless and in lockstep, wait until it answers, and kill it when the
//! [`Doom`] value is dropped - including when the run panics, because a Doom
//! left running holds the port and the next run fails to start for a reason
//! that has nothing to do with the next run.
//!
//! **Nothing here knows an absolute path.** The engine binary and the IWAD are
//! located at run time from flags, then environment, then `PATH` - see
//! [`Paths::resolve`]. A sample that bakes in where a machine keeps its files
//! runs on exactly one machine.
//!
//! Swedish Embedded AB builds the supervision layer around simulators and
//! hardware-in-the-loop rigs - process lifetime, transport, and a reproducible
//! episode - for customers training controllers against them. If your team
//! needs that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Where the two things this sample cannot ship live on THIS machine.
#[derive(Clone, Debug)]
pub struct Paths {
    pub binary: PathBuf,
    pub wad: PathBuf,
}

/// What a missing piece of the environment looks like, with the remedy
/// attached. Rule 5 of `samples/README.md`: say what is missing and leave
/// cleanly.
#[derive(Debug)]
pub struct Missing {
    pub what: &'static str,
    pub remedy: String,
}

impl std::fmt::Display for Missing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n  {}", self.what, self.remedy)
    }
}

impl Paths {
    /// Resolve the engine binary and the IWAD from what the caller typed,
    /// falling back only to `PATH` for the binary.
    ///
    /// No environment variable configures either. A run should be reproducible
    /// from its own command line, and "it works on my machine" is usually an
    /// exported variable three weeks old. `PATH` is the one exception and is
    /// not configuration: it is how every program on a Unix finds another.
    pub fn resolve(binary: Option<&str>, wad: Option<&str>) -> Result<Paths, Missing> {
        let binary = binary
            .map(PathBuf::from)
            .or_else(|| which("restful-doom"))
            .ok_or_else(|| Missing {
                what: "no restful-doom binary",
                remedy: "run ./fetch-data.sh (it prints the flags to use), or pass --doom-bin PATH"
                    .into(),
            })?;
        if !binary.exists() {
            return Err(Missing {
                what: "the restful-doom binary does not exist",
                remedy: format!("checked {}", binary.display()),
            });
        }

        let wad = wad.map(PathBuf::from).ok_or_else(|| Missing {
            what: "no IWAD",
            remedy: "pass --wad PATH; ./fetch-data.sh downloads one and prints the flag".into(),
        })?;
        if !wad.exists() {
            return Err(Missing {
                what: "the IWAD does not exist",
                remedy: format!("checked {}", wad.display()),
            });
        }
        Ok(Paths { binary, wad })
    }
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

/// A port nothing is listening on right now.
///
/// Asked of the OS rather than picked from a range: several runs share a
/// machine (an evaluation sweep runs episodes in parallel), and a fixed port
/// makes the second one fail with "address in use" at a moment that looks like
/// a bug in the sample.
fn free_port() -> std::io::Result<u16> {
    let l = TcpListener::bind("127.0.0.1:0")?;
    let port = l.local_addr()?.port();
    drop(l);
    Ok(port)
}

/// A running game, owned.
pub struct Doom {
    child: Child,
    conn: TcpStream,
    pub port: u16,
    /// Every request and reply, if the caller asked for a transcript. This is
    /// the artifact to read when a policy does something inexplicable: it is
    /// the exact JSON the model saw, in order.
    log: Option<std::fs::File>,
}

/// How to start the game.
#[derive(Clone, Debug)]
pub struct Config {
    pub episode: u32,
    pub map: u32,
    /// A level the engine BUILDS rather than one it loads, named by the
    /// scenario it poses. When set, it decides the episode and map, and the
    /// seed shapes the world rather than only the dice.
    pub scenario: Option<String>,
    /// 0..=4, sk_baby .. sk_nightmare.
    pub skill: u32,
    /// Whether the route may only cross ground the player has seen. The
    /// alternative is a distance field over the whole level, which is a
    /// solved map: it exists as a CONTROL to measure this one against, not as
    /// a way to play.
    pub full_map: bool,
    /// Show the engine's own SDL window. Off by default and normally left off:
    /// the sample draws the framebuffer itself, with the decision overlay on
    /// top, so a second window would show the same game with less in it.
    pub engine_window: bool,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            episode: 1,
            map: 1,
            scenario: None,
            skill: 2,
            engine_window: false,
            full_map: false,
        }
    }
}

impl Doom {
    /// Start the engine. `log` is the JSON transcript of this side of the
    /// conversation; `engine_log` is the engine's OWN output, which is where
    /// its route builder explains itself.
    pub fn start(
        paths: &Paths,
        cfg: &Config,
        log: Option<PathBuf>,
        engine_log: Option<PathBuf>,
    ) -> std::io::Result<Doom> {
        let port = free_port()?;
        let mut cmd = Command::new(&paths.binary);
        cmd.arg("-iwad")
            .arg(&paths.wad)
            .arg("-apiport")
            .arg(port.to_string())
            // The agent is the clock: nothing advances except on /api/step.
            .arg("-apilockstep")
            .arg("-warp")
            .arg(cfg.episode.to_string())
            .arg(cfg.map.to_string())
            // -skill is 1-based on the command line and 0-based in the API.
            .arg("-skill")
            .arg((cfg.skill + 1).to_string())
            .arg("-nosound")
            .arg("-nomusic")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if !cfg.engine_window {
            // -noblit keeps the engine RENDERING into its 320x200 framebuffer
            // (which is what /api/frame reads) while skipping the blit, upscale
            // and present that would put it on a screen. Measured on E1M1 it is
            // most of the cost of a tic, and the sample presents the frame
            // itself anyway.
            cmd.arg("-noblit");
            cmd.env("SDL_VIDEODRIVER", "dummy");
            cmd.env("SDL_AUDIODRIVER", "dummy");
        }
        let mut child = cmd.spawn()?;

        // Drain the engine's own output on a thread. It has to be drained
        // whatever else happens: a full pipe buffer blocks the engine inside a
        // printf in the middle of a tic, and the symptom is a game that
        // freezes after a few hundred steps for no visible reason.
        //
        // Kept rather than dropped when asked for. The engine says a great
        // deal worth reading while it builds its route - which cells it could
        // not reach, what stands between the two halves of a level, how long
        // the grid took - and it is the only account of why a route came out
        // the way it did. Dropping it means every such question has to be
        // asked again by hand, against a level in a state that is no longer
        // the one that produced the answer.
        if let Some(out) = child.stdout.take() {
            let mut sink = engine_log.map(std::fs::File::create).transpose()?;
            std::thread::spawn(move || {
                // Bytes, not lines: the engine prints the WAD's own text, and
                // a line of it that is not UTF-8 must not stop the reader.
                // One that did deadlocked the game a few hundred steps later,
                // with the pipe full and the engine blocked inside a printf.
                let mut r = BufReader::new(out);
                let mut line = Vec::new();
                while r.read_until(b'\n', &mut line).unwrap_or(0) > 0 {
                    if let Some(f) = sink.as_mut() {
                        let _ = f.write_all(&line);
                    }
                    line.clear();
                }
            });
        }

        // Readiness is "the port accepts a connection", not "the engine
        // printed that it is listening". Its stdout is a pipe here, so libc
        // makes it block-buffered and that line does not arrive until several
        // kilobytes later - waiting for it deadlocks until the timeout.
        let conn = connect(port, Duration::from_secs(60), &mut child).inspect_err(|_| {
            let _ = child.kill();
        })?;
        let log = log.map(std::fs::File::create).transpose()?;
        Ok(Doom {
            child,
            conn,
            port,
            log,
        })
    }

    /// One request/response over the kept-alive connection.
    fn call(&mut self, method: &str, path: &str, body: Option<&str>) -> std::io::Result<String> {
        let body = body.unwrap_or("");
        let req = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: keep-alive\r\n\r\n{body}",
            body.len()
        );
        self.conn.write_all(req.as_bytes())?;
        self.conn.flush()?;
        let resp = read_response(&mut self.conn)?;
        if let Some(f) = self.log.as_mut() {
            // One JSON object per line: `jq`-able, and greppable by tic.
            let _ = writeln!(f, "{{\"req\":{{\"method\":\"{method}\",\"path\":\"{path}\",\"body\":{}}},\"resp\":{}}}",
                if body.is_empty() { "null" } else { body }, resp);
        }
        Ok(resp)
    }

    /// Apply `actions` and run exactly `tics` tics. Returns the state after.
    pub fn step(&mut self, actions: &str, tics: u32) -> std::io::Result<String> {
        let body = format!("{{\"tics\":{tics},\"actions\":{actions}}}");
        self.call("POST", "/api/step", Some(&body))
    }

    /// Restart the level. `start_distance`, when set, places the player that
    /// many map units of WALKING from the exit rather than at the level's own
    /// spawn - see `DoomEnv::curriculum`.
    pub fn reset(
        &mut self,
        cfg: &Config,
        seed: u64,
        start_distance: Option<i32>,
    ) -> std::io::Result<String> {
        let start = match start_distance {
            Some(d) => format!(",\"startDistance\":{d}"),
            None => String::new(),
        };
        let where_ = match &cfg.scenario {
            Some(name) => format!("\"scenario\":\"{name}\""),
            None => format!("\"episode\":{},\"map\":{}", cfg.episode, cfg.map),
        };
        let body = format!(
            "{{{where_},\"skill\":{},\"seed\":{},\"mapKnowledge\":\"{}\"{start}}}",
            cfg.skill,
            seed % 65536,
            if cfg.full_map { "full" } else { "seen" }
        );
        self.call("POST", "/api/episode", Some(&body))
    }

    pub fn frame(&mut self) -> std::io::Result<String> {
        self.call("GET", "/api/frame", None)
    }

    pub fn map(&mut self) -> std::io::Result<String> {
        self.call("GET", "/api/map", None)
    }

    /// The route's own working: the player's cell, its eight neighbours, and,
    /// when the player is standing somewhere the distance field never reached,
    /// what lies between here and the part of the level that the route does
    /// understand. Only fetched when a run has gone wrong, since
    /// it costs a round trip and answers a question nobody asks while the
    /// player is making progress.
    pub fn route(&mut self) -> std::io::Result<String> {
        self.call("GET", "/api/route", None)
    }

    /// Hold the state the level is in right now, to come back to.
    ///
    /// Not a replay of the actions that led here: the observation reads which
    /// lines the RENDERER has drawn, and rendering is not part of the
    /// deterministic simulation, so a replayed prefix arrives at the same
    /// player with a different idea of what has been seen. The savegame
    /// format archives line flags, so this does not.
    pub fn snapshot(&mut self) -> std::io::Result<String> {
        self.call("POST", "/api/snapshot", Some("{}"))
    }

    /// Put back what [`Doom::snapshot`] held.
    pub fn restore(&mut self) -> std::io::Result<String> {
        self.call("POST", "/api/snapshot/restore", Some("{}"))
    }

    /// Put a thing on the floor `distance` units away, `bearing` degrees
    /// clockwise from where the player is facing.
    pub fn spawn(&mut self, kind: &str, distance: i32, bearing: i32) -> std::io::Result<String> {
        let body = format!("{{\"type\":\"{kind}\",\"distance\":{distance},\"bearing\":{bearing}}}");
        self.call("POST", "/api/world/objects", Some(&body))
    }
}

impl Drop for Doom {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn connect(port: u16, timeout: Duration, child: &mut Child) -> std::io::Result<TcpStream> {
    let start = Instant::now();
    loop {
        match TcpStream::connect(("127.0.0.1", port)) {
            Ok(s) => {
                // The game is frozen between steps, so a reply arrives only
                // after the engine runs the tics; Nagle would add a round trip
                // to every one of them.
                s.set_nodelay(true)?;
                return Ok(s);
            }
            Err(e) => {
                // An engine that failed on its own - a missing WAD, a bad
                // argument - would otherwise be reported as a timeout, sixty
                // seconds later, with the real message on a stream nobody
                // reads.
                if let Ok(Some(status)) = child.try_wait() {
                    return Err(std::io::Error::other(format!(
                        "the engine exited before opening its API port ({status}); \
                         run it by hand with the same -iwad to see why"
                    )));
                }
                if start.elapsed() > timeout {
                    return Err(e);
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

/// Read one HTTP response and return its body.
///
/// Content-Length is required, which is the whole reason the engine's HTTP
/// layer was rewritten to send one: without it the end of a body is only
/// knowable by the connection closing, and a closed connection per step is a
/// TCP handshake per step.
fn read_response(conn: &mut TcpStream) -> std::io::Result<String> {
    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(i) = find(&buf, b"\r\n\r\n") {
            break i + 4;
        }
        let n = conn.read(&mut chunk)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "the game closed the connection",
            ));
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let len: usize = head
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse().ok())?
        })
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "response has no Content-Length",
            )
        })?;
    while buf.len() < head_end + len {
        let n = conn.read(&mut chunk)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "the game closed the connection mid-body",
            ));
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok(String::from_utf8_lossy(&buf[head_end..head_end + len]).to_string())
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}
