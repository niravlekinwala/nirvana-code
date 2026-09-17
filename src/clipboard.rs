use anyhow::{Context, Result};
use arboard::Clipboard;

pub struct ClipboardHelper;

impl ClipboardHelper {
    pub fn copy_text(text: &str) -> Result<()> {
        let mut clipboard = Clipboard::new().context("Failed to initialize clipboard")?;
        clipboard
            .set_text(text)
            .context("Failed to write to clipboard")?;
        Ok(())
    }
}
