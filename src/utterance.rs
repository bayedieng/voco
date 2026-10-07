//! Speech-probability endpointing with onset confirmation, pre-roll and hysteresis.
//! Non-speech never reaches Parakeet or Enigo; short VAD bursts are discarded.
use std::collections::VecDeque;

use crate::{
    Result,
    audio::SAMPLE_RATE,
    vad::{FRAME_SAMPLES, SpeechDetector},
};

// VAD confidence can lag quiet initial consonants by several hundred milliseconds.
const PRE_ROLL: usize = SAMPLE_RATE / 2;
const ATTACK_SAMPLES: usize = 2 * FRAME_SAMPLES;

#[derive(Clone, Copy)]
pub struct EndpointConfig {
    /// Probability required to start speech. Active speech uses a 0.15 lower threshold.
    pub threshold: f32,
    pub silence_ms: u64,
    pub min_speech_ms: u64,
    pub max_seconds: u64,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            silence_ms: 600,
            min_speech_ms: 96,
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
            return Err("VAD probability threshold must be finite and between 0 and 1".into());
        }
        if !(100..=5000).contains(&self.silence_ms) {
            return Err("silence-ms must be between 100 and 5000".into());
        }
        if !(64..=5000).contains(&self.min_speech_ms) {
            return Err("min-speech-ms must be between 64 and 5000".into());
        }
        if !(1..=60).contains(&self.max_seconds) || self.min_speech_ms >= self.max_seconds * 1000 {
            return Err("max-utterance-secs must be 1..60 and longer than min-speech-ms".into());
        }
        Ok(self)
    }
}

pub struct Segmenter<D: SpeechDetector> {
    detector: D,
    config: EndpointConfig,
    frame: [f32; FRAME_SAMPLES],
    filled: usize,
    pre_roll: VecDeque<f32>,
    active: Vec<f32>,
    attack: usize,
    voiced: usize,
    quiet: usize,
}

impl<D: SpeechDetector> Segmenter<D> {
    pub fn new(config: EndpointConfig, detector: D) -> Result<Self> {
        let config = config.validate()?;
        Ok(Self {
            detector,
            config,
            frame: [0.0; FRAME_SAMPLES],
            filled: 0,
            pre_roll: VecDeque::with_capacity(PRE_ROLL),
            active: Vec::with_capacity(config.max_seconds as usize * SAMPLE_RATE + FRAME_SAMPLES),
            attack: 0,
            voiced: 0,
            quiet: 0,
        })
    }

    pub fn push(&mut self, samples: &[f32], mut emit: impl FnMut(Vec<f32>)) -> Result<()> {
        for &sample in samples {
            self.frame[self.filled] = sample;
            self.filled += 1;
            if self.filled == FRAME_SAMPLES {
                self.filled = 0;
                if let Some(utterance) = self.process_frame(FRAME_SAMPLES)? {
                    emit(utterance);
                }
            }
        }
        Ok(())
    }

    fn process_frame(&mut self, valid: usize) -> Result<Option<Vec<f32>>> {
        // No Mel features needed: Silero consumes normalized mono waveform directly.
        let probability = self.detector.probability(&self.frame)?;
        if self.active.is_empty() {
            for &sample in &self.frame[..valid] {
                if self.pre_roll.len() == PRE_ROLL {
                    self.pre_roll.pop_front();
                }
                self.pre_roll.push_back(sample);
            }
            self.attack = if probability >= self.config.threshold {
                self.attack + valid
            } else {
                0
            };
            if self.attack >= ATTACK_SAMPLES {
                self.active.extend(self.pre_roll.iter());
                self.pre_roll.clear();
                self.voiced = self.attack;
                self.quiet = 0;
            }
        } else {
            self.active.extend_from_slice(&self.frame[..valid]);
            let release_threshold = (self.config.threshold - 0.15).max(0.01);
            if probability >= release_threshold {
                self.voiced += valid;
                self.quiet = 0;
            } else {
                self.quiet += valid;
            }
        }
        let silence_samples = self.config.silence_ms as usize * SAMPLE_RATE / 1000;
        if !self.active.is_empty()
            && (self.quiet >= silence_samples
                || self.active.len() >= self.config.max_seconds as usize * SAMPLE_RATE)
        {
            return Ok(self.take());
        }
        Ok(None)
    }

    /// Evaluate a zero-padded final VAD frame but never append padding to the utterance.
    pub fn finish(&mut self) -> Result<Option<Vec<f32>>> {
        if self.filled > 0 {
            let valid = self.filled;
            self.frame[valid..].fill(0.0);
            self.filled = 0;
            if let Some(audio) = self.process_frame(valid)? {
                return Ok(Some(audio));
            }
        }
        Ok(self.take())
    }

    fn take(&mut self) -> Option<Vec<f32>> {
        let minimum = self.config.min_speech_ms as usize * SAMPLE_RATE / 1000;
        let result = if self.voiced >= minimum {
            // VAD's non-speech decision is not a sample-accurate word boundary.
            // Keep the already-captured hangover: low-energy consonants may live in it.
            Some(self.active.clone())
        } else {
            None
        };
        self.active.clear(); // Retain the large buffer across utterances.
        self.pre_roll.clear();
        self.attack = 0;
        self.voiced = 0;
        self.quiet = 0;
        // Keep VAD context/state: successive utterances belong to one continuous stream.
        result
    }

    pub fn reset(&mut self) {
        self.active.clear();
        self.pre_roll.clear();
        self.attack = 0;
        self.voiced = 0;
        self.quiet = 0;
        self.filled = 0;
        self.detector.reset(); // Only reset the neural history when capture has a gap.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedDetector(f32);
    impl SpeechDetector for FixedDetector {
        fn probability(&mut self, _: &[f32; FRAME_SAMPLES]) -> Result<f32> {
            Ok(self.0)
        }
        fn reset(&mut self) {
            self.0 = 0.0;
        }
    }
    fn feed(
        s: &mut Segmenter<FixedDetector>,
        probability: f32,
        frames: usize,
        out: &mut Vec<Vec<f32>>,
    ) {
        s.detector.0 = probability;
        // High amplitude is deliberately unrelated to the simulated speech probability.
        s.push(&vec![0.5; frames * FRAME_SAMPLES], |audio| out.push(audio))
            .unwrap();
    }

    #[test]
    fn non_speech_and_short_bursts_do_not_emit() {
        let mut s = Segmenter::new(EndpointConfig::default(), FixedDetector(0.0)).unwrap();
        let mut out = Vec::new();
        feed(&mut s, 0.0, 40, &mut out);
        feed(&mut s, 0.9, 2, &mut out);
        feed(&mut s, 0.0, 40, &mut out);
        assert!(out.is_empty());
        assert!(s.finish().unwrap().is_none());
    }

    #[test]
    fn confirmed_speech_emits_once_with_pre_roll_and_tail() {
        let mut s = Segmenter::new(EndpointConfig::default(), FixedDetector(0.0)).unwrap();
        let mut out = Vec::new();
        feed(&mut s, 0.0, 20, &mut out);
        feed(&mut s, 0.9, 12, &mut out);
        feed(&mut s, 0.0, 18, &mut out);
        assert!(out.is_empty()); // 576ms quiet, less than the 600ms hangover.
        feed(&mut s, 0.0, 1, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), PRE_ROLL + (10 + 19) * FRAME_SAMPLES);
        feed(&mut s, 0.0, 40, &mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn hysteresis_avoids_chopping_uncertain_speech() {
        let mut s = Segmenter::new(EndpointConfig::default(), FixedDetector(0.0)).unwrap();
        let mut out = Vec::new();
        feed(&mut s, 0.4, 30, &mut out); // Cannot start speech below 0.5.
        assert!(s.active.is_empty());
        feed(&mut s, 0.9, 8, &mut out);
        feed(&mut s, 0.4, 30, &mut out); // Once active, 0.4 sustains speech.
        assert!(out.is_empty());
        assert!(s.finish().unwrap().is_some());
    }

    #[test]
    fn max_duration_splits_and_shutdown_flushes_partial_frame() {
        let config = EndpointConfig {
            max_seconds: 1,
            ..EndpointConfig::default()
        };
        let mut s = Segmenter::new(config, FixedDetector(0.9)).unwrap();
        let mut out = Vec::new();
        feed(&mut s, 0.9, 44, &mut out);
        assert_eq!(out.len(), 1);
        assert!(out[0].len() <= SAMPLE_RATE + FRAME_SAMPLES);
        s.push(&[0.5; 17], |_| panic!("not a complete frame"))
            .unwrap();
        let remaining = s.finish().unwrap().unwrap();
        assert_eq!(remaining.len(), 12 * FRAME_SAMPLES + 17);
    }

    #[test]
    fn delayed_vad_onset_preserves_all_leading_audio() {
        let mut s = Segmenter::new(EndpointConfig::default(), FixedDetector(0.0)).unwrap();
        let mut out = Vec::new();
        s.push(&vec![0.0; 20 * FRAME_SAMPLES], |_| {}).unwrap();
        // Quiet consonants arrive 288ms before VAD becomes confident.
        s.detector.0 = 0.2;
        s.push(&vec![0.25; 9 * FRAME_SAMPLES], |_| {}).unwrap();
        feed(&mut s, 0.9, 8, &mut out);
        feed(&mut s, 0.0, 19, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].iter().filter(|&&x| x == 0.25).count(),
            9 * FRAME_SAMPLES
        );
    }

    #[test]
    fn quiet_word_endings_are_not_trimmed_out() {
        let mut s = Segmenter::new(EndpointConfig::default(), FixedDetector(0.9)).unwrap();
        let mut out = Vec::new();
        feed(&mut s, 0.9, 8, &mut out);
        s.detector.0 = 0.1;
        s.push(&vec![0.25; 6 * FRAME_SAMPLES], |_| {}).unwrap();
        s.push(&vec![0.0; 13 * FRAME_SAMPLES], |audio| out.push(audio))
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].iter().filter(|&&x| x == 0.25).count(),
            6 * FRAME_SAMPLES
        );
    }

    #[test]
    fn short_confirmed_words_are_not_discarded() {
        let mut s = Segmenter::new(EndpointConfig::default(), FixedDetector(0.9)).unwrap();
        let mut out = Vec::new();
        feed(&mut s, 0.9, 3, &mut out);
        feed(&mut s, 0.0, 19, &mut out);
        assert_eq!(
            out.len(),
            1,
            "96ms of confirmed speech must not be silently discarded"
        );
    }

    #[test]
    fn gaps_discard_active_speech_and_reset_detector() {
        let mut s = Segmenter::new(EndpointConfig::default(), FixedDetector(0.9)).unwrap();
        s.push(&vec![0.1; SAMPLE_RATE], |_| panic!("too early"))
            .unwrap();
        s.reset();
        assert_eq!(s.detector.0, 0.0);
        assert!(s.finish().unwrap().is_none());
    }

    #[test]
    fn detector_failure_propagates_without_emitting() {
        struct Broken;
        impl SpeechDetector for Broken {
            fn probability(&mut self, _: &[f32; FRAME_SAMPLES]) -> Result<f32> {
                Err("VAD failure".into())
            }
            fn reset(&mut self) {}
        }
        let mut s = Segmenter::new(EndpointConfig::default(), Broken).unwrap();
        assert!(
            s.push(&[0.0; FRAME_SAMPLES], |_| panic!("must not emit"))
                .is_err()
        );
    }
}
