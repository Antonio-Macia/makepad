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
//! (see [`decode`]); from then on those modifiers are OR-ed with the real ones
//! until it posts the message again with `0`. Posted messages are retrieved in
//! FIFO order, so "set Ctrl → WM_KEYDOWN Z → WM_KEYUP Z → clear" arrives in that
//! order and the KeyDown carries `control: true`.
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
//! Gate: the channel is OFF unless the process starts with the environment
//! variable [`ENV_VAR`] set to `1`. An app in production never accepts
//! modifiers from another process this way.
//!
//! MUTATIONS checked (2026-10-11, by running them): see the tests below. Making
//! [`decode`] ignore bit 0 → red `decode_reads_each_bit`; making
//! [`suppresses_char`] return false for Ctrl → red
//! `ctrl_drops_printable_chars_shift_does_not`. And the wiring, which no unit
//! test can see, end to end with Brasa's `atajos_sinteticos_demo` driven by
//! `tools/verificacion/pilotar-mensajes.ps1` (window in the background):
//! "abc" + Ctrl+Z gives `[]` with the channel on and `[abcz]` with it off;
//! removing the `suppresses_char` check from `WM_CHAR` gives `[]` then `[z]`.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

/// Name passed to `RegisterWindowMessageW`, by the app and by the pilot.
pub const SYNTHETIC_MODIFIERS_MESSAGE_NAME: &str = "MakepadSyntheticModifiers";

/// Environment variable that turns the channel on (value `1`).
pub const ENV_VAR: &str = "MAKEPAD_SYNTHETIC_MODIFIERS";

/// `wParam` bits.
pub const BIT_CONTROL: u8 = 1;
pub const BIT_SHIFT: u8 = 2;
pub const BIT_ALT: u8 = 4;
pub const BIT_LOGO: u8 = 8;

static ACTIVE: AtomicU8 = AtomicU8::new(0);
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

/// Stores the synthetic bits (only the low four are kept).
pub fn set(bits: u8) {
    ACTIVE.store(bits & 0x0f, Ordering::Relaxed);
}

/// The synthetic bits now in force (`0` when the channel is off or cleared).
pub fn active() -> u8 {
    ACTIVE.load(Ordering::Relaxed)
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
    fn set_keeps_only_the_four_bits() {
        set(0xff);
        assert_eq!(active(), 0x0f);
        set(0);
        assert_eq!(active(), 0);
    }
}
