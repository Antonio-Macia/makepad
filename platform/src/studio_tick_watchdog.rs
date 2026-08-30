//! Studio/embedding frame-clock watchdog — turns the commonest embedding bug
//! from *silence* into *a sentence that names the cause*.
//!
//! # The bug this exists for
//!
//! An app launched with `--stdin-loop` (Studio's run view, a VST3/CLAP plugin
//! panel, any foreign host window) has **no event loop of its own**. Grep
//! `call_draw_event` across `platform/src/os/*/*_stdin.rs`: there is exactly
//! one per platform and every one of them sits inside the
//! `StudioToApp::Tick` branch. `redraw()` and `redraw_all()` only set a dirty
//! flag; **the host owns the frame clock**.
//!
//! So a host that never ticks produces a failure with no error in it: the
//! websocket connects, `CreateWindow` is acknowledged, the widget tree marks
//! itself dirty, and then nothing — forever. Measured on Linux with
//! `tools/studio_embed_host`:
//!
//! ```text
//! --ticks=always  ticks_sent=147  draws_seen=1   (correct host)
//! --ticks=raf     ticks_sent=0    draws_seen=0   (no errors at all)
//! --ticks=none    ticks_sent=0    draws_seen=0   (no errors at all)
//! ```
//!
//! The `raf` row is the one that bites real hosts: a host that sends `Tick`
//! only as an answer to `AppToStudio::RequestAnimationFrame` never starts the
//! cycle, because the app only *asks* for `RequestAnimationFrame` from inside
//! the handling of a `Tick`. Chicken and egg, black panel, clean log.
//!
//! # What this module does
//!
//! A background thread watches two facts the stdin loop reports to it: when a
//! `Tick` arrives, and whether the app currently has drawing pending. If
//! drawing has been pending for longer than the timeout **and no `Tick` has
//! arrived in that window**, it says so once, naming the cause and the fix.
//!
//! It deliberately separates the states that look identical from the app side —
//! nothing is drawn, no error anywhere — and have different fixes:
//!
//! * **never ticked** → the host's tick policy is wrong (the `raf` trap);
//! * **ticked, then stopped** → the host froze, died, or paused the panel;
//! * **ticks fine, never drew once** → the *other* half of the handshake is
//!   missing. This is the one that looks most like a bug in makepad and is not:
//!   the host author checks the tick policy, finds it correct, and has nowhere
//!   left to look. Measured with the harness, same policy, one flag apart:
//!   `--ticks=always` gives 474 ticks and 1 draw; adding `--bootstrap-once`
//!   gives 474 ticks and 0 draws;
//! * **ticks fine, drew before, now stuck** → something wedged after a good
//!   start, so pointing at the handshake would send the reader to code that is
//!   already correct.
//!
//! ⚠ **What this does NOT guard on, and why it matters:** whether a
//! `StudioToApp::Swapchain` ever arrived. That was the first design and it is
//! wrong — a host is not obliged to use makepad's swapchain at all. An embedder
//! presenting through its own shared texture (a VST3 panel on a DXGI handle)
//! never sends that message and is perfectly healthy; guarding on it turns a
//! correct host into a permanent accusation, which is how a guard gets muted.
//! Whether a draw ran is true on every platform and every presentation path.
//!
//! A thread is required rather than a check inside the loop: with no ticks the
//! stdin loop is blocked in `recv_studio_websocket_message` and no code of ours
//! runs at all. That is the whole point — nothing inside the loop can notice
//! its own silence.
//!
//! # Where it is reported
//!
//! Both to `stderr` **and** to the studio log stream. Not belt-and-braces: in
//! studio mode `crate::error!` stops printing locally and travels to the host
//! as an `AppToStudio::LogItem`, so a guard that only used the macro would be
//! delivered to the very host that is misbehaving, and a host that does not
//! surface app logs would swallow the diagnostic about itself.
//!
//! # Tuning
//!
//! `MAKEPAD_STUDIO_TICK_TIMEOUT_S` sets the timeout in seconds; `0` disables
//! the watchdog entirely (for a host that legitimately parks a hidden panel
//! for long stretches).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Default silence tolerated before complaining. Long enough that a slow cold
/// start (shader compilation, font atlas) never trips it, short enough that a
/// developer running a host by hand sees it while still looking at the screen.
const DEFAULT_TIMEOUT_S: f64 = 5.0;

/// Started once per process.
static STARTED: AtomicBool = AtomicBool::new(false);
/// Said its piece once. A watchdog that repeats becomes noise and gets muted.
static REPORTED: AtomicBool = AtomicBool::new(false);

/// Milliseconds since the watchdog started, of the last `Tick` seen.
/// `0` means "no tick has ever arrived", which is a different bug from "the
/// ticks stopped" and is reported differently.
static LAST_TICK_MS: AtomicU64 = AtomicU64::new(0);
/// Ticks seen, for the message. Cheap, and it is the number a host author
/// needs in order to believe the diagnosis.
static TICKS_SEEN: AtomicU64 = AtomicU64::new(0);
/// Milliseconds since the watchdog started, of the moment drawing became
/// pending. `0` means nothing is pending, so there is nothing to complain
/// about: an idle app that is not asking to draw is behaving correctly.
static DRAW_PENDING_SINCE_MS: AtomicU64 = AtomicU64::new(0);
/// Milliseconds of the FIRST tick, so "how long has this host been driving us"
/// can be answered without the pending-draw clock, which is useless here: the
/// Tick branch calls `call_draw_event`, that clears the dirty flag, and the
/// pending clock therefore resets on every single tick.
static FIRST_TICK_MS: AtomicU64 = AtomicU64::new(0);
/// Whether `call_draw_event` has EVER run in this process.
///
/// 🔴 This is the fact that separates the two failures that both look like
/// "ticks flow and the panel is black", and picking it right took two goes.
///
/// The first attempt watched for a `StudioToApp::Swapchain` instead, on the
/// reasoning that without one the repaint skips the pass silently. True, and
/// still the wrong signal: **a host is not obliged to use makepad's swapchain
/// at all.** An embedder that presents through its own shared texture (a VST3
/// panel on a DXGI handle, for one) never sends that message and is perfectly
/// healthy. Guarding on it turns a correct host into a permanent accusation —
/// which is how a guard gets muted and stops protecting anyone.
///
/// Whether a draw ran is true on every platform and every presentation path.
static DREW_EVER: AtomicBool = AtomicBool::new(false);

/// Wall clock shared by the reporting thread and the event loop. `OnceLock`
/// rather than a `SystemTime` so the numbers cannot go backwards on a clock
/// adjustment mid-measurement.
static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn now_ms() -> u64 {
    let epoch = EPOCH.get_or_init(Instant::now);
    // Saturates at 0 only in the first tick of the process; +1 keeps 0 free as
    // the "never happened" sentinel.
    epoch.elapsed().as_millis() as u64 + 1
}

fn timeout() -> Option<Duration> {
    let secs = std::env::var("MAKEPAD_STUDIO_TICK_TIMEOUT_S")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(DEFAULT_TIMEOUT_S);
    if secs <= 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(secs))
}

/// Called by every platform's `stdin_event_loop` right after startup.
/// Idempotent; does nothing if the timeout has been set to zero.
pub fn start_studio_tick_watchdog() {
    if STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let Some(timeout) = timeout() else {
        return;
    };
    EPOCH.get_or_init(Instant::now);
    std::thread::Builder::new()
        .name("studio-tick-watchdog".into())
        .spawn(move || watchdog_main(timeout))
        .ok();
}

/// Called from the `StudioToApp::Tick` branch of the stdin event loop.
pub fn note_studio_tick() {
    TICKS_SEEN.fetch_add(1, Ordering::Relaxed);
    let ahora = now_ms();
    let _ = FIRST_TICK_MS.compare_exchange(0, ahora, Ordering::Relaxed, Ordering::Relaxed);
    LAST_TICK_MS.store(ahora, Ordering::Relaxed);
}

/// Called right after `call_draw_event` runs in the stdin event loop.
///
/// Marks that the app got as far as actually drawing at least once. Before
/// that, no number of ticks can produce anything.
pub fn note_studio_drew() {
    DREW_EVER.store(true, Ordering::Relaxed);
}

/// Called after each host message with the app's current dirty state, i.e.
/// `self.need_redrawing()`.
///
/// Passing `true` starts (or keeps) the clock on "this app is waiting for a
/// frame it cannot produce by itself"; passing `false` stops it, because an
/// app with nothing to draw is not a symptom of anything.
pub fn note_studio_draw_pending(pending: bool) {
    if pending {
        // Only the FIRST moment counts; re-stamping on every message would
        // keep pushing the deadline away and the guard would never fire.
        let _ = DRAW_PENDING_SINCE_MS.compare_exchange(
            0,
            now_ms(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    } else {
        DRAW_PENDING_SINCE_MS.store(0, Ordering::Relaxed);
    }
}

/// What the watchdog concluded from the three numbers it watches.
///
/// Three states on purpose, not two: "the host never ticked" and "the host
/// ticked and then stopped" look identical from inside the app — nothing is
/// drawn either way — and they have different fixes (a wrong tick policy
/// versus a host that froze, died or parked the panel). Collapsing them would
/// send half the readers to the wrong place.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum Verdict {
    /// Nothing to say: either nothing is waiting to be drawn, or not enough
    /// time has passed, or the host is ticking normally.
    Quiet,
    /// Drawing has been pending past the timeout and no tick has EVER arrived.
    NeverTicked,
    /// Ticks arrived at some point and then stopped, and drawing is pending.
    TicksStopped,
    /// 🔴 Ticks ARE arriving and this app has never drawn once.
    ///
    /// This is the failure that looks most like a bug in makepad and is not:
    /// the host does the tick part right, so every obvious check passes, and
    /// the panel still never paints. The cause is almost always the other half
    /// of the handshake — the bootstrap (`WindowGeomChange` + `Swapchain`) sent
    /// once and lost to a cold start, after which the host ticks forever into
    /// an app that never learned where to draw.
    ///
    /// ⚠ And it cannot be caught with the pending-draw clock, which is why it
    /// went unnoticed: the Tick branch calls `call_draw_event`, the dirty flag
    /// clears, and the app looks idle and healthy on every single tick.
    ///
    /// Measured with `tools/studio_embed_host`, same policy, one flag apart:
    ///
    /// ```text
    /// --ticks=always                   ticks_sent=415  draws_seen=1
    /// --ticks=always --bootstrap-once  ticks_sent=414  draws_seen=0
    /// ```
    TicksButNeverDrew,
    /// Ticks are arriving, the app HAS drawn before, and drawing has still been
    /// pending past the timeout. Something wedged after a good start — a draw
    /// that hangs, or a swapchain dropped on a resize and never replaced.
    TicksButFrameStuck,
}

/// The whole decision, as a pure function of the four numbers, so it can be
/// exercised without a process, a host or a clock.
///
/// * `pending_since_ms` — when drawing became pending; `0` = nothing pending.
/// * `last_tick_ms` — when the last tick arrived; `0` = never.
/// * `first_tick_ms` — when the first tick arrived; `0` = never.
/// * `drew_ever` — whether `call_draw_event` has ever run.
pub(crate) fn verdict(
    now_ms: u64,
    pending_since_ms: u64,
    last_tick_ms: u64,
    timeout_ms: u64,
    first_tick_ms: u64,
    drew_ever: bool,
) -> Verdict {
    // 🔴 Checked FIRST, and before the pending-draw clock, because this case is
    // invisible to that clock: with ticks flowing, `call_draw_event` runs and
    // clears the dirty flag on every tick, so `pending_since_ms` is 0 almost
    // always and every later test would return Quiet. Ordering this after them
    // is exactly how the commonest embedding failure went unreported.
    if first_tick_ms != 0
        && !drew_ever
        && now_ms.saturating_sub(first_tick_ms) >= timeout_ms
    {
        return Verdict::TicksButNeverDrew;
    }
    if pending_since_ms == 0 {
        return Verdict::Quiet;
    }
    if now_ms.saturating_sub(pending_since_ms) < timeout_ms {
        return Verdict::Quiet;
    }
    // A tick inside the waiting window means the host IS driving us. That used
    // to return Quiet — "the fault is elsewhere, not our call to make" — and
    // that was wrong: if drawing has been pending past the timeout WHILE ticks
    // flow, the app is being driven, wants to draw, and still cannot. By this
    // point the swapchain case is already ruled out above, so what is left is a
    // good start that wedged.
    if last_tick_ms >= pending_since_ms && last_tick_ms != 0 {
        return Verdict::TicksButFrameStuck;
    }
    if last_tick_ms == 0 {
        Verdict::NeverTicked
    } else {
        Verdict::TicksStopped
    }
}

fn watchdog_main(timeout: Duration) {
    let timeout_ms = timeout.as_millis() as u64;
    loop {
        std::thread::sleep(Duration::from_millis(250));
        if REPORTED.load(Ordering::Relaxed) {
            return;
        }
        let pending_since = DRAW_PENDING_SINCE_MS.load(Ordering::Relaxed);
        let last_tick = LAST_TICK_MS.load(Ordering::Relaxed);
        let first_tick = FIRST_TICK_MS.load(Ordering::Relaxed);
        let drew = DREW_EVER.load(Ordering::Relaxed);
        let now = now_ms();
        let v = verdict(
            now,
            pending_since,
            last_tick,
            timeout_ms,
            first_tick,
            drew,
        );
        if v == Verdict::Quiet {
            continue;
        }
        if REPORTED.swap(true, Ordering::SeqCst) {
            return;
        }
        // For the swapchain case the pending clock is meaningless (it resets on
        // every tick), so the number that means something is how long the host
        // has been ticking us.
        let esperado = if v == Verdict::TicksButNeverDrew {
            now.saturating_sub(first_tick)
        } else {
            now.saturating_sub(pending_since)
        };
        report(v, esperado, last_tick);
        return;
    }
}

fn report(verdict: Verdict, waited_ms: u64, last_tick_ms: u64) {
    let ticks = TICKS_SEEN.load(Ordering::Relaxed);
    let waited = waited_ms as f64 / 1000.0;

    // The two "ticks are flowing" verdicts get their own message. They share no
    // fix with the tick-policy ones, and sending their reader through a
    // paragraph about RequestAnimationFrame — when their host already ticks
    // correctly — is how a guard earns the reputation of crying wolf.
    if matches!(
        verdict,
        Verdict::TicksButNeverDrew | Verdict::TicksButFrameStuck
    ) {
        let message = if verdict == Verdict::TicksButNeverDrew {
            format!(
                "makepad embedding: {ticks} StudioToApp::Tick received over {waited:.1}s and \
                 this app has never DRAWN once -- call_draw_event has not run a single time, \
                 so the frame clock is reaching us and there is still nothing to present. \
                 That is the other half of the handshake missing, not the clock. The \
                 frame clock is fine, so the missing half is the rest of the handshake: the \
                 host must send the bootstrap (StudioToApp::WindowGeomChange AND \
                 StudioToApp::Swapchain) and, critically, KEEP RE-SENDING IT until the first \
                 frame lands. Sending it once is the natural thing to write -- it is \
                 handshake, not per-frame data -- and it loses the race on a cold start: the \
                 app is still starting when the only copy arrives, and from then on the host \
                 ticks forever into an app that never learned where to draw. The reference \
                 host (studio/desktop/src/desktop_run_view.rs) re-sends every 15 ticks until \
                 the first successful present. Reproduce both sides with \
                 `tools/studio_embed_host --ticks=always [--bootstrap-once]`. Set \
                 MAKEPAD_STUDIO_TICK_TIMEOUT_S=0 to silence this check."
            )
        } else {
            format!(
                "makepad embedding: {ticks} StudioToApp::Tick received and nothing has been \
                 drawn for {waited:.1}s, and this app HAS drawn before. So the handshake \
                 worked and something wedged afterwards -- most often a swapchain \
                 dropped on a resize and never replaced (re-send StudioToApp::Swapchain \
                 after every WindowGeomChange), or a draw that is blocking. Set \
                 MAKEPAD_STUDIO_TICK_TIMEOUT_S=0 to silence this check."
            )
        };
        emitir(message);
        return;
    }

    let diagnosis = if last_tick_ms == 0 {
        "this app has NEVER received a StudioToApp::Tick".to_string()
    } else {
        let since = (now_ms().saturating_sub(last_tick_ms)) as f64 / 1000.0;
        format!(
            "the last StudioToApp::Tick arrived {since:.1}s ago (\
             {ticks} received in total), and the ticks then stopped"
        )
    };

    let message = format!(
        "makepad embedding: nothing has been drawn for {waited:.1}s and {diagnosis}. \
         In --stdin-loop (studio/embedded) mode the ONLY place that turns a redraw into a \
         frame is the StudioToApp::Tick branch of the stdin event loop: redraw() and \
         redraw_all() by themselves never produce a frame, so with no Tick there is no \
         frame, ever. Your host must send StudioToApp::Tick UNCONDITIONALLY on every one \
         of its own frames -- NOT only as a reply to AppToStudio::RequestAnimationFrame, \
         because the app only asks for RequestAnimationFrame from INSIDE the handling of a \
         Tick, so a reply-only host never starts the cycle. The reference host \
         (studio/desktop/src/desktop_run_view.rs) also re-sends its bootstrap \
         (WindowGeomChange + Swapchain) every 15 ticks until the first successful present, \
         which is what survives a cold start. Set MAKEPAD_STUDIO_TICK_TIMEOUT_S=0 to \
         silence this check."
    );

    emitir(message);
}

/// Say it, on both channels.
///
/// Printed to stderr as well as through `error!` on purpose: with the studio
/// socket connected, `error!` stops printing locally and ships the text to the
/// host -- the same host this message is accusing. A diagnosis delivered only
/// to the party at fault is a diagnosis nobody reads.
fn emitir(message: String) {
    eprintln!("[makepad] ERROR {message}");
    crate::error!("{}", message);
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT: u64 = 5_000;
    /// "The app has drawn before" — the normal state, so the interesting tests
    /// are the ones that pass `false`.
    const YA_DIBUJO: bool = true;
    /// No tick has ever arrived. Keeps the swapchain check out of the way in
    /// the tests that are about the tick clock.
    const SIN_TICKS: u64 = 0;

    #[test]
    fn an_idle_app_is_never_accused() {
        // Nothing pending: an app that is not asking to draw is behaving.
        assert_eq!(verdict(100_000, 0, 0, TIMEOUT, SIN_TICKS, YA_DIBUJO), Verdict::Quiet);
        assert_eq!(verdict(100_000, 0, 99_000, TIMEOUT, 99_000, YA_DIBUJO), Verdict::Quiet);
    }

    #[test]
    fn waiting_less_than_the_timeout_is_not_a_fault() {
        // A cold start with shaders and font atlases takes a while; the guard
        // must not shout over it.
        assert_eq!(verdict(4_000, 1_000, 0, TIMEOUT, SIN_TICKS, YA_DIBUJO), Verdict::Quiet);
    }

    #[test]
    fn ticks_flowing_and_never_a_draw_is_the_lost_bootstrap() {
        // 🔴 This case used to return Quiet, on the reasoning that a host which
        // ticks is driving us correctly so the fault must be someone else's.
        // That was wrong, and it was the WORST place to be wrong: it is the
        // failure that looks most like a bug in makepad, because every obvious
        // check the host author performs comes back green.
        //
        // Measured with tools/studio_embed_host, same tick policy, one flag
        // apart: 415 ticks and 1 frame with the bootstrap re-sent, 414 ticks
        // and 0 frames with it sent once.
        assert_eq!(
            verdict(100_000, 1_000, 99_000, TIMEOUT, 900, false),
            Verdict::TicksButNeverDrew
        );
    }

    #[test]
    fn ticks_flowing_after_good_frames_is_a_different_verdict() {
        // Same silence as above, opposite cause: the handshake DID work, so
        // pointing the reader at the bootstrap would send them to look at code
        // that is already correct. Here it is a swapchain lost on a resize, or
        // a draw that blocks.
        assert_eq!(
            verdict(100_000, 1_000, 99_000, TIMEOUT, 900, YA_DIBUJO),
            Verdict::TicksButFrameStuck
        );
    }

    #[test]
    fn a_host_that_never_ticked_is_named_as_such() {
        // The reply-only host: it waits for RequestAnimationFrame, which the
        // app only sends from inside a Tick it will never get.
        assert_eq!(verdict(10_000, 1_000, 0, TIMEOUT, SIN_TICKS, YA_DIBUJO), Verdict::NeverTicked);
    }

    #[test]
    fn a_host_that_stopped_ticking_is_a_different_verdict() {
        // Ticks up to 900ms, then the panel is resized at 1_000 and the clock
        // never comes back. Same silence, different fix.
        assert_eq!(
            verdict(10_000, 1_000, 900, TIMEOUT, 900, YA_DIBUJO),
            Verdict::TicksStopped
        );
    }

    #[test]
    fn the_boundary_belongs_to_the_timeout_not_to_the_alarm() {
        // Exactly at the timeout it fires; one millisecond short it does not.
        assert_eq!(verdict(6_000, 1_000, 0, TIMEOUT, SIN_TICKS, YA_DIBUJO), Verdict::NeverTicked);
        assert_eq!(verdict(5_999, 1_000, 0, TIMEOUT, SIN_TICKS, YA_DIBUJO), Verdict::Quiet);
    }

    #[test]
    fn every_verdict_is_reachable() {
        // 🔴 A guard whose green cannot be turned red measures nothing. This
        // pins that each of the four states has at least one input producing
        // it, so a future refactor that makes one unreachable fails here
        // instead of going quiet in production.
        let todos = [
            verdict(100_000, 0, 0, TIMEOUT, SIN_TICKS, YA_DIBUJO),
            verdict(10_000, 1_000, 0, TIMEOUT, SIN_TICKS, YA_DIBUJO),
            verdict(10_000, 1_000, 900, TIMEOUT, 900, YA_DIBUJO),
            verdict(100_000, 1_000, 99_000, TIMEOUT, 900, false),
            verdict(100_000, 1_000, 99_000, TIMEOUT, 900, YA_DIBUJO),
        ];
        for esperado in [
            Verdict::Quiet,
            Verdict::NeverTicked,
            Verdict::TicksStopped,
            Verdict::TicksButNeverDrew,
            Verdict::TicksButFrameStuck,
        ] {
            assert!(
                todos.contains(&esperado),
                "ningun caso produce {esperado:?}: ese veredicto es codigo muerto"
            );
        }
    }
}
