//! Dictation output is kept separate from audio and inference.
use enigo::{Enigo, Keyboard, Settings};

use crate::Result;

#[cfg(target_os = "linux")]
#[path = "output_xwayland.rs"]
mod xwayland;

pub struct DictationOutput {
    keyboard: Option<Enigo>,
    log_transcripts: bool,
    #[cfg(target_os = "linux")]
    xwayland: Option<xwayland::XwaylandKeys>,
}

impl DictationOutput {
    pub fn new(print_only: bool, log_transcripts: bool) -> Result<Self> {
        let keyboard = if print_only {
            None
        } else {
            Some(Enigo::new(&Settings::default())?)
        };
        Ok(Self {
            keyboard,
            log_transcripts: log_transcripts || print_only,
            #[cfg(target_os = "linux")]
            xwayland: if print_only {
                None
            } else {
                xwayland::XwaylandKeys::detect()?
            },
        })
    }

    pub fn emit(&mut self, mut text: String) -> Result<()> {
        if self.log_transcripts {
            println!("{text}");
        }
        if let Some(keyboard) = &mut self.keyboard {
            // Dictate plain text only; never synthesize Enter or other control keys.
            text = text.replace(['\n', '\r', '\t'], " ");
            text.push(' ');
            #[cfg(target_os = "linux")]
            if let Some(keys) = &self.xwayland {
                return keys.text(keyboard, &text);
            }
            keyboard.text(&text)?;
        }
        Ok(())
    }
}
