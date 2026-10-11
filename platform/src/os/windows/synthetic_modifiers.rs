//! Test-only channel to deliver Ctrl/Shift/Alt/Win with posted keyboard messages.
//!
//! Why this exists: `Win32Window::get_key_modifiers` reads the modifiers with
//! `GetKeyState` when a message is processed, not from the message itself. A UI
//! pilot that drives the app with `PostMessageW` (so it works while the app is in
//! the background, which Windows requires: a background process cannot take the
//! foreground to use `SendInput`/`keybd_event`) can type letters and click, but it
//! can never press Ctrl+Z, because posting `WM_KEYDOWN` does not change the key
//! state. Measured in Pregón on 2026-10-04 (Brasa matrix row C110): undo/redo were
//! verified through the buttons, the shortcut never.
//!
//! How it works: the pilot posts the registered message
//! [`SYNTHETIC_MODIFIERS_MESSAGE_NAME`] to the window with a bitmask in `wParam`
//! (see [`decode`]); from then on those modifiers are OR-ed with the real ones.
//! Posted messages are retrieved in FIFO order, so "set Ctrl → WM_KEYDOWN Z →
//! WM_KEYUP Z → clear" arrives in that order and the KeyDown carries
//! `control: true`.
//!
//! They can NOT stay stuck, and that is a design rule, not a nicety. If the pilot
//! died between "set" and "clear", a process-wide Ctrl left on would make the app
//! unusable for the person in front of it: no typing, V pastes the real
//! clipboard, X cuts the selection, and with Alt, F4 closes the window. Clearing
//! on focus loss is no remedy, because the window is in the background on
//! purpose. So the bits live for ONE key press and are cleared by whichever comes
//! first: the pilot's `0`, the `WM_KEYUP` of a key that is not a modifier
//! ([`releases_on_key_up`]), or [`LIFETIME_MS`] after they were set
//! ([`in_force`]). The pilot's explicit `0` stays the normal path; the other two
//! are the net for when it never arrives.
//!
//! The `WM_CHAR` trap: the message loop runs `TranslateMessage`, which builds the
//! character from the REAL key state. A posted `Z` with a synthetic Ctrl would
//! produce `'z'` (and type it into a focused text field) where a real Ctrl+Z
//! produces the control character 0x1A, which the `WM_CHAR` handler drops. So
//! while Ctrl, Alt or Win are synthetic, printable `WM_CHAR`s are dropped too
//! ([`suppresses_char`]). Shift alone does not drop them: Shift+letter is text.
//! Its imperfection, stated: with a synthetic Shift the translated char is still
//! lowercase. Pilots type text with `WM_CHAR` directly, so that is not a problem.
//!
//! Gate: the window procedure only listens to the message when the process
//! starts with the environment variable [`ENV_VAR`] set to `1`; without it
//! [`message_id`] is `0` and no other process can turn the modifiers on. Two
//! limits, stated: the variable is inherited by child processes like any other
//! (a pilot that starts the app should set it on that child only, not on its own
//! shell); and inside the process [`set`] is `pub(crate)`, so only the platform
//! layer can call it, but it does not check the variable itself.
//!
//! MUTATIONS checked (2026-10-11, by running them): see the tests below. Making
//! [`decode`] ignore bit 0 → red `decode_reads_each_bit`; making
//! [`suppresses_char`] return false for Ctrl → red
//! `ctrl_drops_printable_chars_shift_does_not`; making [`in_force`] ignore the
//! age → red `the_bits_expire`; making [`releases_on_key_up`] return true for a
//! modifier → red `a_modifier_key_up_does_not_clear`. And the wiring, which no unit
//! test can see, end to end with Brasa's `atajos_sinteticos_demo` driven by
//! `tools/verificacion/pilotar-mensajes.ps1` (window in the background):
//! "abc" + Ctrl+Z gives `[]` with the channel on and `[abcz]` with it off;
//! removing the `suppresses_char` check from `WM_CHAR` gives `[]` then `[z]`.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Name passed to `RegisterWindowMessageW`, by the app and by the pilot.
pub const SYNTHETIC_MODIFIERS_MESSAGE_NAME: &str = "MakepadSyntheticModifiers";

/// Environment variable that turns the channel on (value `1`).
pub const ENV_VAR: &str = "MAKEPAD_SYNTHETIC_MODIFIERS";

/// `wParam` bits.
pub const BIT_CONTROL: u8 = 1;
pub const BIT_SHIFT: u8 = 2;
pub const BIT_ALT: u8 = 4;
pub const BIT_LOGO: u8 = 8;

/// How long the bits stay in force after being set, in milliseconds. A pilot
/// posts "set → down → up → clear" in well under a millisecond, and the clock
/// starts when the window procedure PROCESSES "set", so a busy app does not eat
/// into it; two seconds only bound how long a dead pilot can leave them on.
pub const LIFETIME_MS: u64 = 2_000;

static ACTIVE: AtomicU8 = AtomicU8::new(0);
/// When [`ACTIVE`] was last set, in ms since [`clock_ms`]'s origin.
static SET_AT_MS: AtomicU64 = AtomicU64::new(0);
static MESSAGE_ID: OnceLock<u32> = OnceLock::new();

/// Modifiers decoded from a bitmask: `(control, shift, alt, logo)`.
pub fn decode(bits: u8) -> (bool, bool, bool, bool) {
    (
        bits & BIT_CONTROL != 0,
        bits & BIT_SHIFT != 0,
        bits & BIT_ALT != 0,
        bits & BIT_LOGO != 0,
    )
}

/// Whether a printable `WM_CHAR` must be dropped with these synthetic bits.
pub fn suppresses_char(bits: u8) -> bool {
    bits & (BIT_CONTROL | BIT_ALT | BIT_LOGO) != 0
}

/// The bits still in force: `bits` if they were set less than [`LIFETIME_MS`]
/// ago, `0` otherwise. Pure, so the expiry is testable without a clock.
pub fn in_force(bits: u8, set_at_ms: u64, now_ms: u64) -> u8 {
    if now_ms.saturating_sub(set_at_ms) < LIFETIME_MS {
        bits
    } else {
        0
    }
}

/// Whether releasing this key ends the synthetic press. A modifier's own
/// `WM_KEYUP` does not (a pilot may post Ctrl's key messages too); any other
/// key does, because the shortcut it was part of is over.
pub fn releases_on_key_up(key_is_modifier: bool) -> bool {
    !key_is_modifier
}

/// The registered message id, or `0` when the channel is off.
///
/// Registered once; `0` is never a valid registered message, so matching
/// `msg == message_id()` with the channel off can never fire.
pub fn message_id() -> u32 {
    *MESSAGE_ID.get_or_init(|| {
        if std::env::var(ENV_VAR).as_deref() != Ok("1") {
            return 0;
        }
        let name: Vec<u16> = SYNTHETIC_MODIFIERS_MESSAGE_NAME
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        // The vendored `windows` crate is trimmed and does not carry this
        // binding, so it is declared here with the same macro it uses.
        windows_core::link!("user32.dll" "system" fn RegisterWindowMessageW(lpstring: *const u16) -> u32);
        unsafe { RegisterWindowMessageW(name.as_ptr()) }
    })
}

/// Monotonic milliseconds since the first call in this process.
fn clock_ms() -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Stores the synthetic bits (only the low four are kept) and starts their
/// [`LIFETIME_MS`]. Crate-only: the window procedure calls it after checking
/// the message id, which is where the environment gate lives.
pub(crate) fn set(bits: u8) {
    SET_AT_MS.store(clock_ms(), Ordering::Relaxed);
    ACTIVE.store(bits & 0x0f, Ordering::Relaxed);
}

/// Called on every `WM_KEYUP`: clears the bits when the released key ends the
/// synthetic press (see [`releases_on_key_up`]).
pub(crate) fn key_released(key_is_modifier: bool) {
    if releases_on_key_up(key_is_modifier) {
        ACTIVE.store(0, Ordering::Relaxed);
    }
}

/// The synthetic bits now in force (`0` when the channel is off, cleared or
/// expired).
pub fn active() -> u8 {
    in_force(
        ACTIVE.load(Ordering::Relaxed),
        SET_AT_MS.load(Ordering::Relaxed),
        clock_ms(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_reads_each_bit() {
        assert_eq!(decode(0), (false, false, false, false));
        assert_eq!(decode(BIT_CONTROL), (true, false, false, false));
        assert_eq!(decode(BIT_SHIFT), (false, true, false, false));
        assert_eq!(decode(BIT_ALT), (false, false, true, false));
        assert_eq!(decode(BIT_LOGO), (false, false, false, true));
        assert_eq!(decode(BIT_CONTROL | BIT_SHIFT), (true, true, false, false));
    }

    #[test]
    fn ctrl_drops_printable_chars_shift_does_not() {
        assert!(suppresses_char(BIT_CONTROL));
        assert!(suppresses_char(BIT_ALT));
        assert!(suppresses_char(BIT_LOGO));
        assert!(suppresses_char(BIT_CONTROL | BIT_SHIFT));
        assert!(!suppresses_char(BIT_SHIFT));
        assert!(!suppresses_char(0));
    }

    #[test]
    fn the_bits_expire() {
        assert_eq!(in_force(BIT_CONTROL, 100, 100), BIT_CONTROL);
        assert_eq!(in_force(BIT_CONTROL, 100, 100 + LIFETIME_MS - 1), BIT_CONTROL);
        assert_eq!(in_force(BIT_CONTROL, 100, 100 + LIFETIME_MS), 0);
        // A clock that went backwards (it cannot, it is monotonic) keeps them.
        assert_eq!(in_force(BIT_CONTROL, 100, 50), BIT_CONTROL);
    }

    #[test]
    fn a_modifier_key_up_does_not_clear() {
        assert!(!releases_on_key_up(true));
        assert!(releases_on_key_up(false));
    }

    #[test]
    fn set_keeps_only_the_four_bits() {
        set(0xff);
        assert_eq!(active(), 0x0f);
        key_released(true);
        assert_eq!(active(), 0x0f, "a modifier's key-up keeps them");
        key_released(false);
        assert_eq!(active(), 0, "any other key-up ends the press");
        set(BIT_CONTROL);
        set(0);
        assert_eq!(active(), 0);
    }
}
