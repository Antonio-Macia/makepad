use crate::event::SafeAreaInsets;
use crate::Vec2d;

const DEFAULT_MIN_DESKTOP_WIDTH: f64 = 860.;

/// Controls how the system bars (status bar and navigation bar) icons and
/// text are tinted, on platforms that support it (currently Android and iOS;
/// iOS only has a status bar).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SystemBarAppearance {
    /// Pick dark or light system-bar icons automatically based on the
    /// luminance of the window's background color: a light background gets
    /// dark icons, a dark background gets light icons. This is the default.
    #[default]
    Auto,
    /// Force dark icons/text in the system bars (best for light backgrounds).
    DarkIcons,
    /// Force light icons/text in the system bars (best for dark backgrounds).
    LightIcons,
}

/// The current context data relevant to adaptive views.
/// Later to be expanded with more context data like platfrom information, accessibility settings, etc.
///
/// NOTE: `Default` is implemented by hand, not derived. See `text_scale`.
#[derive(Clone, Debug)]
pub struct DisplayContext {
    /// The event ID that last updated the display context
    pub updated_on_event_id: u64,
    /// The current screen size
    pub screen_size: Vec2d,
    /// Safe area insets for the current window in Makepad layout points
    /// (non-zero on devices with notches, rounded corners, home indicators, etc.)
    pub safe_area_insets: SafeAreaInsets,
    /// Controls the tint of the system bar (status/navigation bar) icons.
    /// Set via [`crate::Cx::set_system_bar_appearance`]; resolved and applied
    /// by the `Window` widget.
    pub system_bar_appearance: SystemBarAppearance,
    /// The text size multiplier the user asked for in the OS accessibility
    /// settings. `1.0` is normal; Android allows up to `2.0`.
    ///
    /// This is deliberately SEPARATE from screen density, and that separation is
    /// the whole point. An OS exposes two different settings -- "display size"
    /// and "font size" -- because they solve two different problems: the first
    /// is how much information fits, the second is whether anyone can READ it.
    /// Someone who raises the second does it because of their eyesight, and
    /// scaling the layout instead would just move the same problem around.
    ///
    /// Makepad had no platform-independent way to reach this value, so every app
    /// built with it silently ignored the setting: raising it changed nothing.
    /// Widgets that lay text out should multiply their font size by this.
    ///
    /// Defaults to `1.0` on platforms that don't report it. NEVER let this reach
    /// `0.0` -- that scales every glyph to nothing and the UI goes blank with no
    /// error anywhere. That is why `Default` is written out below instead of
    /// derived: a derived `Default` would put `0.0` here.
    pub text_scale: f64,
}

impl Default for DisplayContext {
    fn default() -> Self {
        Self {
            updated_on_event_id: 0,
            screen_size: Vec2d::default(),
            safe_area_insets: SafeAreaInsets::default(),
            system_bar_appearance: SystemBarAppearance::default(),
            // The neutral multiplier, not zero. See the field docs.
            text_scale: 1.0,
        }
    }
}

impl DisplayContext {
    pub fn is_desktop(&self) -> bool {
        self.screen_size.x >= DEFAULT_MIN_DESKTOP_WIDTH
    }

    /// Whether the given width qualifies as the wide "desktop" layout.
    /// Useful as a fallback signal when `screen_size` isn't known yet.
    pub fn is_desktop_width(&self, width: f64) -> bool {
        width >= DEFAULT_MIN_DESKTOP_WIDTH
    }

    pub fn is_screen_size_known(&self) -> bool {
        self.screen_size.x != 0.0 && self.screen_size.y != 0.0
    }
}
