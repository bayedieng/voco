//! Lightweight RMS-based endpointing, not a neural speech detector.
//! All decisions use 20ms frames at 16kHz, independent of CPAL callback sizes.
use std::collections::VecDeque;

use crate::{Result, audio::SAMPLE_RATE};

const FRAME: usize = SAMPLE_RATE / 50;
const PRE_ROLL: usize = SAMPLE_RATE / 5;
const ATTACK_FRAMES: usize = 3;
const KEEP_SILENCE: usize = SAMPLE_RATE / 10;

#[derive(Clone, Copy)]
pub struct EndpointConfig {
    pub threshold: f32,
    pub silence_ms: u64,
    pub min_speech_ms: u64,
    pub max_seconds: u64,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            threshold: 0.01,
            silence_ms: 600,
            min_speech_ms: 200,
            max_seconds: 15,
        }
    }
}

impl EndpointConfig {
    pub fn validate(self) -> Result<Self> {
        if !self.threshold.is_finite()
            || !(0.0..1.0).contains(&self.threshold)
            || self.threshold == 0.0
        {
            return Err("threshold must be finite and between 0 and 1".into());
        }
        if !(100..=5000).contains(&self.silence_ms) {
            return Err("silence-ms must be between 100 and 5000".into());
        }
        if !(60..=5000).contains(&self.min_speech_ms) {
            return Err("min-speech-ms must be between 60 and 5000".into());
        }
        if !(1..=60).contains(&self.max_seconds) || self.min_speech_ms >= self.max_seconds * 1000 {
            return Err("max-utterance-secs must be 1..60 and longer than min-speech-ms".into());
        }
        Ok(self)
    }
}

pub struct Segmenter {
    config: EndpointConfig,
    frame: [f32; FRAME],
    filled: usize,
    pre_roll: VecDeque<f32>,
    active: Vec<f32>,
    attack: usize,
    voiced: usize,
    quiet: usize,
}

impl Segmenter {
    pub fn new(config: EndpointConfig) -> Result<Self> {
        let config = config.validate()?;
        Ok(Self {
            config,
            frame: [0.0; FRAME],
            filled: 0,
            pre_roll: VecDeque::with_capacity(PRE_ROLL),
            active: Vec::with_capacity(config.max_seconds as usize * SAMPLE_RATE),
            attack: 0,
            voiced: 0,
            quiet: 0,
        })
    }

    pub fn push(&mut self, samples: &[f32], mut emit: impl FnMut(Vec<f32>)) {
        for &sample in samples {
            self.frame[self.filled] = sample;
            self.filled += 1;
            if self.filled == FRAME {
                self.filled = 0;
                if let Some(utterance) = self.process_frame() {
                    emit(utterance);
                }
            }
        }
    }

    fn process_frame(&mut self) -> Option<Vec<f32>> {
        let power = self.frame.iter().map(|x| x * x).sum::<f32>() / FRAME as f32;
        let speech = power >= self.config.threshold * self.config.threshold;
        if self.active.is_empty() {
            for &sample in &self.frame {
                if self.pre_roll.len() == PRE_ROLL {
                    self.pre_roll.pop_front();
                }
                self.pre_roll.push_back(sample);
            }
            self.attack = if speech { self.attack + 1 } else { 0 };
            if self.attack >= ATTACK_FRAMES {
                self.active.extend(self.pre_roll.iter());
                self.pre_roll.clear();
                self.voiced = self.attack;
                self.quiet = 0;
            }
        } else {
            self.active.extend_from_slice(&self.frame);
            if speech {
                self.voiced += 1;
                self.quiet = 0;
            } else {
                self.quiet += 1;
            }
        }
        let silence_frames = self.config.silence_ms.div_ceil(20) as usize;
        if !self.active.is_empty()
            && (self.quiet >= silence_frames
                || self.active.len() >= self.config.max_seconds as usize * SAMPLE_RATE)
        {
            return self.take();
        }
        None
    }

    /// Flush a valid unfinished utterance on Ctrl+C.
    pub fn finish(&mut self) -> Option<Vec<f32>> {
        if !self.active.is_empty() && self.filled > 0 {
            self.active.extend_from_slice(&self.frame[..self.filled]);
        }
        self.filled = 0;
        self.take()
    }

    fn take(&mut self) -> Option<Vec<f32>> {
        let result = if self.voiced * 20 >= self.config.min_speech_ms as usize {
            let trim = (self.quiet * FRAME).saturating_sub(KEEP_SILENCE);
            Some(self.active[..self.active.len().saturating_sub(trim)].to_vec())
        } else {
            None
        };
        self.active.clear(); // Preserve the large allocation across utterances.
        self.pre_roll.clear();
        self.attack = 0;
        self.voiced = 0;
        self.quiet = 0;
        result
    }

    /// Never join speech across a capture gap or inject an incomplete transcription.
    pub fn reset(&mut self) {
        self.active.clear();
        self.pre_roll.clear();
        self.attack = 0;
        self.voiced = 0;
        self.quiet = 0;
        self.filled = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(segmenter: &mut Segmenter, value: f32, ms: usize, output: &mut Vec<Vec<f32>>) {
        segmenter.push(&vec![value; ms * SAMPLE_RATE / 1000], |audio| {
            output.push(audio)
        });
    }

    #[test]
    fn silence_and_clicks_do_not_emit() {
        let mut s = Segmenter::new(EndpointConfig::default()).unwrap();
        let mut out = Vec::new();
        feed(&mut s, 0.0, 1000, &mut out);
        feed(&mut s, 0.1, 80, &mut out);
        feed(&mut s, 0.0, 1000, &mut out);
        assert!(out.is_empty());
        assert!(s.finish().is_none());
    }

    #[test]
    fn speech_emits_once_after_silence_with_pre_roll() {
        let mut s = Segmenter::new(EndpointConfig::default()).unwrap();
        let mut out = Vec::new();
        feed(&mut s, 0.0, 500, &mut out);
        feed(&mut s, 0.1, 400, &mut out);
        feed(&mut s, 0.0, 580, &mut out);
        assert!(out.is_empty());
        feed(&mut s, 0.0, 20, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), (140 + 400 + 100) * SAMPLE_RATE / 1000);
        feed(&mut s, 0.0, 1000, &mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn max_duration_splits_and_shutdown_flushes() {
        let config = EndpointConfig {
            max_seconds: 1,
            ..EndpointConfig::default()
        };
        let mut s = Segmenter::new(config).unwrap();
        let mut out = Vec::new();
        feed(&mut s, 0.1, 1400, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), SAMPLE_RATE);
        assert!(s.finish().is_some());
    }

    #[test]
    fn gaps_discard_active_speech() {
        let mut s = Segmenter::new(EndpointConfig::default()).unwrap();
        s.push(&vec![0.1; SAMPLE_RATE], |_| panic!("too early"));
        s.reset();
        assert!(s.finish().is_none());
    }
}
