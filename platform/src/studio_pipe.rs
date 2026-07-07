//! Local-pipe transport for the studio protocol (`--stdin-loop` mode).
//!
//! By default an app running in `--stdin-loop` mode talks to its host
//! (Makepad Studio) over a websocket. That is the right transport for the
//! IDE, but a poor fit when the host is another local process embedding the
//! app's rendered output — e.g. an audio-plugin shell that composites the
//! shared swapchain texture inside a DAW window. For that use case a local
//! IPC endpoint (a Unix domain socket on unix, a named pipe on Windows) is
//! lower latency, needs no TCP port, and is private to the session.
//!
//! This module adds that transport **additively**: when the environment
//! variable [`MAKEPAD_STUDIO_PIPE`] is set to an endpoint path, the app
//! connects to it instead of opening the studio websocket. Nothing changes
//! for the websocket path when the variable is absent.
//!
//! ### Wire format
//! The payloads are exactly the same bincode blobs the websocket binary
//! path uses ([`AppToStudioVec`] / `StudioToAppVec` via `SerBin`), framed
//! with a little-endian `u32` length prefix:
//!
//! ```text
//! [u32-le payload_len][payload_len bytes of SerBin data]  (repeat)
//! ```
//!
//! Both directions use the same framing. A frame larger than
//! [`MAX_FRAME_LEN`] is treated as a protocol error and closes the
//! connection (protects against desync producing a bogus huge allocation).
//!
//! ### Roles
//! The **host** (plugin shell / embedder) creates the endpoint as a server
//! *before* spawning the app process; the **app** (this side) connects as a
//! client, retrying for a few seconds to tolerate startup races.
//!
//! ### Integration points (all in `web_socket.rs` / `app_main` flow)
//! - `Cx::init_websockets` first calls [`init_studio_pipe_from_env`]; when
//!   it returns `true` the websocket machinery is skipped entirely.
//! - `Cx::send_studio_message` routes through [`send_binary`] in pipe mode.
//! - `Cx::recv_studio_websocket_message` routes through [`recv_timeout`]
//!   in pipe mode (see `recv_studio_pipe_message`), so the per-platform
//!   stdin event loops need no changes at all.

use crate::makepad_network::WebSocketMessage;
use std::{
    io::{Read, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{channel, Receiver, RecvTimeoutError, Sender},
        Mutex,
    },
    time::{Duration, Instant},
};

/// Environment variable holding the endpoint path the app must connect to.
/// Unix: a filesystem path to a Unix domain socket.
/// Windows: a named-pipe path such as `\\.\pipe\my-app-ui-1234`.
pub const MAKEPAD_STUDIO_PIPE: &str = "MAKEPAD_STUDIO_PIPE";

/// Upper bound for a single frame. The studio protocol ships input events,
/// geometry and occasional screenshots/widget dumps; 64 MiB is far above
/// anything legitimate while still catching framing desync early.
pub const MAX_FRAME_LEN: u32 = 64 * 1024 * 1024;

/// How long the client keeps retrying to connect at startup. The host
/// creates the endpoint before spawning us, but process startup order can
/// still race on a loaded machine.
const CONNECT_DEADLINE: Duration = Duration::from_secs(10);
const CONNECT_RETRY_SLEEP: Duration = Duration::from_millis(20);

/// True when the pipe transport is active (env var was set and the initial
/// connection succeeded).
static STUDIO_PIPE_MODE: AtomicBool = AtomicBool::new(false);

/// Write half of the connection. Boxed so unix/windows streams share one
/// static. Mutex-serialised: writers are the UI thread plus (rarely) log
/// forwarding from other threads.
static STUDIO_PIPE_WRITER: Mutex<Option<Box<dyn Write + Send>>> = Mutex::new(None);

/// Frames parsed by the reader thread, consumed by the stdin event loop.
static STUDIO_PIPE_RX: Mutex<Option<Receiver<WebSocketMessage>>> = Mutex::new(None);

/// Result of one [`recv_timeout`] poll, mirroring what the event loop needs
/// to distinguish: data, "nothing yet, go service other work", or gone.
pub(crate) enum PipeRecv {
    Msg(WebSocketMessage),
    Timeout,
    Closed,
}

/// Is the pipe transport active for this process?
pub fn studio_pipe_mode() -> bool {
    STUDIO_PIPE_MODE.load(Ordering::SeqCst)
}

/// Connect to the endpoint named by [`MAKEPAD_STUDIO_PIPE`], if set.
///
/// Returns `true` when the transport is up (caller must then skip the
/// websocket init). Returns `false` when the variable is unset, on
/// unsupported platforms, or if the connection could not be established
/// within [`CONNECT_DEADLINE`] (an error is logged; falling back to the
/// websocket would hide a broken host setup, so we do NOT fall back —
/// the app keeps running standalone without a studio link).
pub fn init_studio_pipe_from_env() -> bool {
    let Ok(path) = std::env::var(MAKEPAD_STUDIO_PIPE) else {
        return false;
    };
    let path = path.trim().to_string();
    if path.is_empty() {
        return false;
    }
    match connect_with_retry(&path) {
        Ok((reader, writer)) => {
            let (tx, rx) = channel();
            *STUDIO_PIPE_WRITER.lock().unwrap() = Some(writer);
            *STUDIO_PIPE_RX.lock().unwrap() = Some(rx);
            STUDIO_PIPE_MODE.store(true, Ordering::SeqCst);
            // Keep the internal "studio is attached" paths (screenshots,
            // widget dumps, profiling, the stdin loop's transport guard)
            // active, exactly like set_studio_stdout_mode does.
            crate::web_socket::HAS_STUDIO_WEB_SOCKET.store(true, Ordering::SeqCst);
            crate::web_socket::STUDIO_WEB_SOCKET_CONNECTED.store(true, Ordering::SeqCst);
            spawn_reader_thread(reader, tx);
            crate::log!("studio pipe transport active: {}", path);
            true
        }
        Err(err) => {
            crate::error!("could not connect studio pipe {}: {}", path, err);
            false
        }
    }
}

/// Send one already-serialised `AppToStudioVec` payload, length-prefixed.
/// On write failure the connection is torn down (host most likely died);
/// the reader thread will surface `Closed` to the event loop.
pub(crate) fn send_binary(data: Vec<u8>) {
    let mut writer = STUDIO_PIPE_WRITER.lock().unwrap();
    let Some(w) = writer.as_mut() else {
        return;
    };
    let len = (data.len() as u32).to_le_bytes();
    let result = w
        .write_all(&len)
        .and_then(|_| w.write_all(&data))
        .and_then(|_| w.flush());
    if let Err(err) = result {
        crate::error!("studio pipe write failed, closing: {}", err);
        *writer = None;
        crate::web_socket::STUDIO_WEB_SOCKET_CONNECTED.store(false, Ordering::SeqCst);
    }
}

/// Poll the reader channel with a timeout. `Closed` is terminal: it is
/// returned when the reader thread has exited (EOF, IO error or protocol
/// error) and its channel is dropped.
pub(crate) fn recv_timeout(timeout: Duration) -> PipeRecv {
    let rx = STUDIO_PIPE_RX.lock().unwrap();
    let Some(rx) = rx.as_ref() else {
        return PipeRecv::Closed;
    };
    match rx.recv_timeout(timeout) {
        Ok(msg) => PipeRecv::Msg(msg),
        Err(RecvTimeoutError::Timeout) => PipeRecv::Timeout,
        Err(RecvTimeoutError::Disconnected) => PipeRecv::Closed,
    }
}

/// Reader thread: parse `[u32-le len][payload]` frames into
/// `WebSocketMessage::Binary` (the same envelope the websocket path
/// produces, so downstream deserialisation is shared). Exits — dropping
/// the sender, which surfaces `Closed` — on EOF, IO error, or an
/// over-limit frame.
fn spawn_reader_thread(mut reader: Box<dyn Read + Send>, tx: Sender<WebSocketMessage>) {
    std::thread::Builder::new()
        .name("studio-pipe-reader".into())
        .spawn(move || loop {
            let mut len_buf = [0u8; 4];
            if reader.read_exact(&mut len_buf).is_err() {
                // EOF or error: host went away. Dropping tx signals Closed.
                return;
            }
            let len = u32::from_le_bytes(len_buf);
            if len == 0 {
                continue; // empty keep-alive frame, ignore
            }
            if len > MAX_FRAME_LEN {
                crate::error!("studio pipe frame too large ({} bytes), closing", len);
                return;
            }
            let mut payload = vec![0u8; len as usize];
            if reader.read_exact(&mut payload).is_err() {
                return;
            }
            if tx.send(WebSocketMessage::Binary(payload)).is_err() {
                return; // event loop gone
            }
        })
        .expect("failed to spawn studio-pipe-reader thread");
}

/// Platform connection: returns independent read/write halves.
#[cfg(unix)]
fn connect_with_retry(path: &str) -> std::io::Result<(Box<dyn Read + Send>, Box<dyn Write + Send>)> {
    use std::os::unix::net::UnixStream;
    let deadline = Instant::now() + CONNECT_DEADLINE;
    loop {
        match UnixStream::connect(path) {
            Ok(stream) => {
                let reader = stream.try_clone()?;
                return Ok((Box::new(reader), Box::new(stream)));
            }
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::NotFound
                        | std::io::ErrorKind::ConnectionRefused
                        | std::io::ErrorKind::Interrupted
                ) && Instant::now() < deadline =>
            {
                std::thread::sleep(CONNECT_RETRY_SLEEP);
            }
            Err(err) => return Err(err),
        }
    }
}

/// Windows named pipes accept plain `CreateFile` opens: `File` with
/// read+write on `\\.\pipe\<name>` is a byte-mode client connection.
/// `try_clone` duplicates the handle for the reader thread.
#[cfg(windows)]
fn connect_with_retry(path: &str) -> std::io::Result<(Box<dyn Read + Send>, Box<dyn Write + Send>)> {
    use std::fs::OpenOptions;
    let deadline = Instant::now() + CONNECT_DEADLINE;
    loop {
        match OpenOptions::new().read(true).write(true).open(path) {
            Ok(file) => {
                let reader = file.try_clone()?;
                return Ok((Box::new(reader), Box::new(file)));
            }
            // NotFound: pipe not created yet. PermissionDenied can transiently
            // surface as ERROR_PIPE_BUSY while the host is between
            // ConnectNamedPipe calls; retry both until the deadline.
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
                ) && Instant::now() < deadline =>
            {
                std::thread::sleep(CONNECT_RETRY_SLEEP);
            }
            Err(err) => return Err(err),
        }
    }
}

/// Everything else (wasm, mobile): the pipe transport does not apply.
#[cfg(not(any(unix, windows)))]
fn connect_with_retry(_path: &str) -> std::io::Result<(Box<dyn Read + Send>, Box<dyn Write + Send>)> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "studio pipe transport is only supported on unix and windows",
    ))
}
