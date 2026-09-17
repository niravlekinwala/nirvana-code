use ratatui::style::{Color, Modifier, Style};

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct Theme {
    pub bg_dark: Color,
    pub bg_card: Color,
    pub bg_input: Color,
    pub border_dim: Color,
    pub border_focus: Color,
    pub border_cyan: Color,
    pub neon_cyan: Color,
    pub neon_magenta: Color,
    pub neon_amber: Color,
    pub neon_green: Color,
    pub neon_purple: Color,
    pub text_bright: Color,
    pub text_dim: Color,
    pub text_muted: Color,
    pub badge_bg: Color,
    pub badge_fg: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            bg_dark: Color::Rgb(10, 12, 16),
            bg_card: Color::Rgb(16, 20, 26),
            bg_input: Color::Rgb(22, 28, 38),
            border_dim: Color::Rgb(38, 48, 64),
            border_focus: Color::Rgb(0, 230, 255),
            border_cyan: Color::Rgb(0, 210, 255),
            neon_cyan: Color::Rgb(0, 240, 255),
            neon_magenta: Color::Rgb(255, 0, 128),
            neon_amber: Color::Rgb(255, 180, 0),
            neon_green: Color::Rgb(0, 255, 150),
            neon_purple: Color::Rgb(180, 100, 255),
            text_bright: Color::Rgb(240, 245, 255),
            text_dim: Color::Rgb(170, 185, 205),
            text_muted: Color::Rgb(90, 105, 125),
            badge_bg: Color::Rgb(24, 34, 52),
            badge_fg: Color::Rgb(0, 230, 255),
        }
    }
}

impl Theme {
    pub fn title_style(&self) -> Style {
        Style::default()
            .fg(self.neon_cyan)
            .add_modifier(Modifier::BOLD)
    }

    pub fn badge_style(&self) -> Style {
        Style::default()
            .fg(self.badge_fg)
            .bg(self.badge_bg)
            .add_modifier(Modifier::BOLD)
    }

    pub fn green_badge(&self) -> Style {
        Style::default()
            .fg(self.neon_green)
            .bg(Color::Rgb(10, 32, 24))
            .add_modifier(Modifier::BOLD)
    }

    pub fn amber_badge(&self) -> Style {
        Style::default()
            .fg(self.neon_amber)
            .bg(Color::Rgb(36, 28, 10))
            .add_modifier(Modifier::BOLD)
    }

    pub fn magenta_badge(&self) -> Style {
        Style::default()
            .fg(self.neon_magenta)
            .bg(Color::Rgb(36, 10, 24))
            .add_modifier(Modifier::BOLD)
    }

    pub fn border_style(&self, active: bool) -> Style {
        if active {
            Style::default().fg(self.border_focus)
        } else {
            Style::default().fg(self.border_dim)
        }
    }
}
