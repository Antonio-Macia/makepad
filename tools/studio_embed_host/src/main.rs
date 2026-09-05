//! `studio_embed_host` — the smallest thing that can host an embedded makepad
//! app, plus a switch to make it host it *wrongly* on purpose.
//!
//! # Why this exists
//!
//! An app started with `--stdin-loop` (Studio's run view, a DAW plugin panel,
//! any foreign window) has no event loop of its own. Look at
//! `platform/src/os/*/‌*_stdin.rs`: the one and only `call_draw_event` on that
//! path lives inside the `StudioToApp::Tick` branch. Everything else —
//! `redraw()`, `redraw_all()`, a widget marking itself dirty — only sets a
//! flag. **The host owns the frame clock.**
//!
//! That makes the commonest embedding bug completely silent: a host that sends
//! `Tick` only as a reply to `AppToStudio::RequestAnimationFrame` never starts
//! the cycle, because the app only asks for `RequestAnimationFrame` from
//! *inside* the handling of a `Tick`. Chicken and egg, no error, black panel.
//!
//! So this host speaks the real protocol over the real websocket and can be
//! told which kind of host to be:
//!
//! | `--ticks=` | Behaviour | Expected outcome |
//! |---|---|---|
//! | `always` | `Tick` on every host frame, unconditionally | app draws (this is the correct host) |
//! | `raf`    | `Tick` only in reply to `RequestAnimationFrame` | app never draws — the real-world bug |
//! | `none`   | never sends `Tick` | app never draws |
//!
//! And one flag that is not about the clock at all, because the clock is only
//! half of the contract:
//!
//! | flag | Behaviour | Expected outcome |
//! |---|---|---|
//! | `--bootstrap-once` | ticks correctly, but sends the geometry handshake a single time | app never draws — and every obvious check on the host comes back green |
//!
//! That last row is the one worth having. Measured, same policy, one flag
//! apart:
//!
//! ```text
//! --ticks=always                   ticks_sent=474  draws_seen=1
//! --ticks=always --bootstrap-once  ticks_sent=474  draws_seen=0
//! ```
//!
//! Sending the handshake once is the natural thing to write — it is handshake,
//! not per-frame data — and it works on every warm run and loses the race on a
//! cold one. The reference host re-sends every 15 ticks until the first frame
//! lands, and that is what survives a cold start.
//!
//! One asymmetry worth knowing before reading a `raf` run: only
//! `windows_stdin.rs`, `macos_stdin.rs` and the headless loop ever emit
//! `AppToStudio::RequestAnimationFrame` — the Linux x11 stdin loop never sends
//! it at all. So on Linux `--ticks=raf` and `--ticks=none` are the same host in
//! practice; the chicken-and-egg trap in its pure form is Windows/macOS shaped.
//! The cause being measured — the one `call_draw_event` living inside `Tick` —
//! is identical on all four.
//!
//! # What it does *not* do
//!
//! It does not allocate a swapchain, so nothing is ever presented. That is
//! deliberate: the question here is "did `call_draw_event` run?", and the probe
//! answers it by logging. Pixels need the shared-framebuffer path (DXGI handle
//! / IOSurface / dma-buf) which is platform-specific and irrelevant to the
//! frame-clock contract this tool measures.
//!
//! # Usage
//!
//! ```text
//! cargo run -p makepad-studio-embed-host -- \
//!     --ticks=always --port=8099 --seconds=6 -- \
//!     ./target/debug/makepad-studio-embed-probe
//! ```
//!
//! The child is spawned with `STUDIO=http://127.0.0.1:<port>/app?build=1&crate=<name>`
//! and `--stdin-loop`, which is exactly how Studio launches a build.

use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Serve the auxiliary channel an embedded app opens on Linux.
///
/// This one is not optional and it is not obvious. On Linux the app answers
/// `StudioToApp::WindowGeomChange` by calling
/// `aux_chan::ClientEndpoint::connect_from_studio_env()`, which **blocks the
/// whole stdin event loop for ten seconds** retrying a unix socket at
/// `/tmp/makepad-stdin-aux-<port>-<build>.sock` before giving up with a log
/// line. A host that does not create that socket therefore freezes its own
/// embedded UI for ten seconds per geometry change, and the only visible
/// symptom is that nothing happens.
///
/// We do not need the file descriptors it carries (no swapchain here), only
/// the accept, so the app can carry on.
#[cfg(unix)]
fn serve_aux_channel(port: u16, build_id: &str) {
    use std::os::unix::net::UnixListener;
    let path = format!("/tmp/makepad-stdin-aux-{port}-{build_id}.sock");
    let _ = std::fs::remove_file(&path);
    let Ok(listener) = UnixListener::bind(&path) else {
        println!("[host] could not bind aux channel at {path}");
        return;
    };
    std::thread::spawn(move || {
        // Held forever: closing an accepted endpoint would make the app treat
        // the channel as gone.
        let mut kept = Vec::new();
        for stream in listener.incoming().flatten() {
            println!("[host] aux channel client accepted");
            kept.push(stream);
        }
    });
}

#[cfg(not(unix))]
fn serve_aux_channel(_port: u16, _build_id: &str) {}

use makepad_micro_serde::{DeBin, SerBin};
use makepad_network::http_server::{start_http_server, HttpServer, HttpServerRequest};
use makepad_studio_protocol::{AppToStudio, AppToStudioVec, StudioToApp, StudioToAppVec};

/// How the host decides to send `StudioToApp::Tick`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum TickPolicy {
    /// Correct: a tick on every host frame, whether or not the app asked.
    Always,
    /// Broken, and the one that bites in the wild: a tick only as an answer to
    /// `AppToStudio::RequestAnimationFrame`.
    OnRequestAnimationFrame,
    /// Broken and obvious: no ticks at all.
    Never,
}

impl TickPolicy {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "always" => Some(Self::Always),
            "raf" | "on-raf" => Some(Self::OnRequestAnimationFrame),
            "none" | "never" => Some(Self::Never),
            _ => None,
        }
    }
}

struct Args {
    ticks: TickPolicy,
    port: u16,
    seconds: u64,
    /// Frames per second of the simulated host clock.
    host_fps: u64,
    /// Stop ticking after this many seconds — simulates a host that froze,
    /// died, or parked the panel. Looks the same from inside the app as a host
    /// that never ticked, and needs a different fix, so it is worth being able
    /// to produce it on purpose.
    stop_ticks_after: Option<u64>,
    /// Send the bootstrap (geometry + swapchain) exactly ONCE instead of
    /// re-sending it until the first frame lands.
    ///
    /// Reproduces the failure of a host that does the tick part right and still
    /// never gets a frame — the one that looks like a makepad bug and is not.
    bootstrap_once: bool,
    child: Vec<String>,
}

fn parse_args() -> Args {
    let mut ticks = TickPolicy::Always;
    let mut port = 8099u16;
    let mut seconds = 6u64;
    let mut host_fps = 60u64;
    let mut stop_ticks_after = None;
    let mut bootstrap_once = false;
    let mut child = Vec::new();
    let mut in_child = false;
    for arg in std::env::args().skip(1) {
        if in_child {
            child.push(arg);
            continue;
        }
        if arg == "--" {
            in_child = true;
        } else if let Some(v) = arg.strip_prefix("--ticks=") {
            ticks = TickPolicy::parse(v).unwrap_or_else(|| panic!("bad --ticks={v}"));
        } else if let Some(v) = arg.strip_prefix("--port=") {
            port = v.parse().expect("bad --port");
        } else if let Some(v) = arg.strip_prefix("--seconds=") {
            seconds = v.parse().expect("bad --seconds");
        } else if let Some(v) = arg.strip_prefix("--fps=") {
            host_fps = v.parse::<u64>().expect("bad --fps").max(1);
        } else if let Some(v) = arg.strip_prefix("--stop-ticks-after=") {
            stop_ticks_after = Some(v.parse::<u64>().expect("bad --stop-ticks-after"));
        } else if arg == "--bootstrap-once" {
            bootstrap_once = true;
        } else {
            panic!("unknown argument {arg}");
        }
    }
    assert!(!child.is_empty(), "pass the child binary after `--`");
    Args {
        ticks,
        port,
        seconds,
        host_fps,
        stop_ticks_after,
        bootstrap_once,
        child,
    }
}

/// Everything the websocket writer thread and the message loop share.
#[derive(Default)]
struct Shared {
    /// Framed-message sink for the single connected app, once it connects.
    to_app: Option<mpsc::Sender<Vec<u8>>>,
    /// Windows the app has asked us to create, in the order it asked.
    windows: Vec<usize>,
    /// `RequestAnimationFrame` messages the app has sent us and we have not
    /// answered yet (only consulted under `--ticks=raf`).
    pending_raf: u64,
    /// Ticks this host has actually PUT ON THE WIRE. Counted here, on the
    /// sending side, because a count taken inside the app cannot tell "the
    /// host sent none" from "the host sent some and they were lost" — both
    /// read as zero.
    ticks_sent: u64,
    /// `PROBE-DRAW` lines received from the app.
    draws_seen: u64,
    /// Bootstrap re-sends, the trick the real host uses to survive a cold
    /// start (see `studio/desktop/src/desktop_run_view.rs`).
    bootstraps_sent: u64,
    app_connected: bool,
}

fn send(shared: &mut Shared, msgs: Vec<StudioToApp>) {
    if msgs.is_empty() {
        return;
    }
    let ticks = msgs
        .iter()
        .filter(|m| matches!(m, StudioToApp::Tick))
        .count() as u64;
    if let Some(tx) = &shared.to_app {
        if tx.send(StudioToAppVec(msgs).serialize_bin()).is_ok() {
            shared.ticks_sent += ticks;
        }
    }
}

/// The window geometry handshake. The real host also ships a `Swapchain` here;
/// we stop one step short on purpose (see the module docs).
fn bootstrap_msgs(shared: &Shared) -> Vec<StudioToApp> {
    shared
        .windows
        .iter()
        .map(|window_id| StudioToApp::WindowGeomChange {
            dpi_factor: 1.0,
            window_id: *window_id,
            left: 0.0,
            top: 0.0,
            width: 420.0,
            height: 220.0,
        })
        .collect()
}

fn main() {
    let args = parse_args();
    let addr = format!("127.0.0.1:{}", args.port)
        .parse()
        .expect("bad listen address");

    let (tx_request, rx_request) = mpsc::channel::<HttpServerRequest>();
    start_http_server(HttpServer {
        listen_address: addr,
        request: tx_request,
        post_max_size: 1024 * 1024,
    })
    .expect("could not bind the host http server");

    let shared = Arc::new(Mutex::new(Shared::default()));

    serve_aux_channel(args.port, "1");

    // The child is launched exactly the way Studio launches a build: STUDIO
    // points at us, and `--stdin-loop` selects the windowless event loop.
    let studio = format!(
        "http://127.0.0.1:{}/app?build=1&crate=makepad-studio-embed-probe",
        args.port
    );
    let mut child: Child = Command::new(&args.child[0])
        .args(&args.child[1..])
        .arg("--stdin-loop")
        .env("STUDIO", &studio)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("could not spawn the embedded app");

    println!("[host] policy={:?} studio={}", args.ticks, studio);

    // The host frame clock. A real DAW ticks this from its own paint loop.
    {
        let shared = shared.clone();
        let policy = args.ticks;
        let args_bootstrap_once = args.bootstrap_once;
        let frame = Duration::from_micros(1_000_000 / args.host_fps);
        let stop_at = args
            .stop_ticks_after
            .map(|secs| Instant::now() + Duration::from_secs(secs));
        let mut nudged = false;
        std::thread::spawn(move || loop {
            std::thread::sleep(frame);
            if stop_at.is_some_and(|stop_at| Instant::now() >= stop_at) {
                // One resize AFTER the clock stopped: it dirties the tree, so
                // the app is now waiting for a frame that will never come.
                // This is what a user does when a frozen plugin panel looks
                // wrong — drag its corner — and it is the state that must be
                // told apart from "the host never ticked at all".
                if !nudged {
                    nudged = true;
                    let mut s = shared.lock().unwrap();
                    let msgs = s
                        .windows
                        .iter()
                        .map(|window_id| StudioToApp::WindowGeomChange {
                            dpi_factor: 1.0,
                            window_id: *window_id,
                            left: 0.0,
                            top: 0.0,
                            width: 640.0,
                            height: 480.0,
                        })
                        .collect();
                    send(&mut s, msgs);
                    println!("[host] ticks stopped; sent one resize to dirty the tree");
                }
                continue;
            }
            let mut s = shared.lock().unwrap();
            if !s.app_connected {
                continue;
            }
            let mut msgs = Vec::new();

            // Cold-start insurance, copied from the reference host: keep
            // re-sending the bootstrap until the app has clearly come alive.
            // Under `always` this is what rescues a handshake that raced the
            // app's own startup.
            if s.draws_seen == 0 && policy == TickPolicy::Always {
                s.bootstraps_sent += 1;
                // 🔴 `--bootstrap-once` reproduces the failure of a host that
                // ticks correctly and STILL never gets a frame, which is the one
                // that looks like a makepad bug and is not.
                //
                // Sending the bootstrap a single time is the obvious thing to
                // write: the geometry and the swapchain are handshake, not
                // per-frame data, so re-sending them looks like waste. It works
                // on every warm run, and loses the race on a cold one — the app
                // is still in startup when the only copy arrives, and after that
                // the host ticks forever into an app that never learned where to
                // draw. Ticks flow, the log is clean, the panel stays black.
                let reenviar = if args_bootstrap_once {
                    s.bootstraps_sent == 1
                } else {
                    s.bootstraps_sent == 1 || s.bootstraps_sent % 15 == 0
                };
                if reenviar {
                    msgs.extend(bootstrap_msgs(&s));
                }
            }

            match policy {
                TickPolicy::Always => msgs.push(StudioToApp::Tick),
                TickPolicy::OnRequestAnimationFrame => {
                    if s.pending_raf > 0 {
                        s.pending_raf -= 1;
                        msgs.push(StudioToApp::Tick);
                    }
                }
                TickPolicy::Never => {}
            }
            send(&mut s, msgs);
        });
    }

    let deadline = Instant::now() + Duration::from_secs(args.seconds);
    while Instant::now() < deadline {
        let Ok(request) = rx_request.recv_timeout(Duration::from_millis(100)) else {
            continue;
        };
        match request {
            HttpServerRequest::ConnectWebSocket {
                response_sender, ..
            } => {
                let mut s = shared.lock().unwrap();
                s.to_app = Some(response_sender);
                s.app_connected = true;
                println!("[host] app websocket connected");
            }
            HttpServerRequest::DisconnectWebSocket { .. } => {
                let mut s = shared.lock().unwrap();
                s.app_connected = false;
                println!("[host] app websocket disconnected");
            }
            HttpServerRequest::BinaryMessage { data, .. } => {
                let Ok(msgs) = AppToStudioVec::deserialize_bin(&data) else {
                    println!("[host] undecodable binary payload ({} bytes)", data.len());
                    continue;
                };
                let mut s = shared.lock().unwrap();
                for msg in msgs.0 {
                    handle_app_msg(&mut s, msg);
                }
            }
            _ => {}
        }
    }

    let s = shared.lock().unwrap();
    println!(
        "[host] RESULT policy={:?} ticks_sent={} draws_seen={} bootstraps={}",
        args.ticks, s.ticks_sent, s.draws_seen, s.bootstraps_sent
    );
    drop(s);
    let _ = child.kill();
    let _ = child.wait();
}

fn handle_app_msg(s: &mut Shared, msg: AppToStudio) {
    match msg {
        AppToStudio::BeforeStartup => println!("[app] BeforeStartup"),
        AppToStudio::AfterStartup => println!("[app] AfterStartup"),
        AppToStudio::CreateWindow { window_id, .. } => {
            println!("[app] CreateWindow {window_id}");
            if !s.windows.contains(&window_id) {
                s.windows.push(window_id);
            }
            let msgs = bootstrap_msgs(s);
            send(s, msgs);
        }
        AppToStudio::RequestAnimationFrame => {
            s.pending_raf += 1;
            println!("[app] RequestAnimationFrame (pending={})", s.pending_raf);
        }
        AppToStudio::LogItem(item) => {
            if item.message.contains("PROBE-DRAW") {
                s.draws_seen += 1;
            }
            println!("[app] log: {}", item.message.trim_end());
        }
        AppToStudio::DrawCompleteAndFlip(_) => println!("[app] DrawCompleteAndFlip"),
        other => println!("[app] {other:?}"),
    }
}
