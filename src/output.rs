//! Dictation output is kept separate from audio and inference.
use enigo::{Enigo, Keyboard, Settings};

use crate::Result;

pub struct DictationOutput {
    keyboard: Option<Enigo>,
}

impl DictationOutput {
    pub fn new(print_only: bool) -> Result<Self> {
        let keyboard = if print_only {
            None
        } else {
            Some(Enigo::new(&Settings::default())?)
        };
        Ok(Self { keyboard })
    }

    pub fn emit(&mut self, mut text: String) -> Result<()> {
        println!("{text}");
        if let Some(keyboard) = &mut self.keyboard {
            // Dictate plain text only; never synthesize Enter or other control keys.
            text = text.replace(['\n', '\r', '\t'], " ");
            text.push(' ');
            keyboard.text(&text)?;
        }
        Ok(())
    }
}
