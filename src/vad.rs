//! Silero VAD v6.2.3: normalized 16kHz audio, 512 new samples + 64 context samples.
//! A single CPU thread and fixed-size reusable buffers keep per-frame overhead small.
use std::path::Path;

use ort::{session::Session, value::TensorRef};

use crate::Result;

pub const FRAME_SAMPLES: usize = 512;
const CONTEXT: usize = 64;
const STATE_SIZE: usize = 2 * 128;
pub const DEFAULT_VAD_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero-vad-v6.2.3.onnx");

/// Separate probability inference from endpointing so boundary logic is deterministic/testable.
pub trait SpeechDetector {
    fn probability(&mut self, frame: &[f32; FRAME_SAMPLES]) -> Result<f32>;
    fn reset(&mut self);
}

pub struct SileroVad {
    session: Session,
    input: [f32; CONTEXT + FRAME_SAMPLES],
    state: [f32; STATE_SIZE],
}

impl SileroVad {
    pub fn load(path: &Path) -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(1)
            .map_err(ort::Error::<()>::from)?
            .with_inter_threads(1)
            .map_err(ort::Error::<()>::from)?
            .commit_from_file(path)?;
        Ok(Self {
            session,
            input: [0.0; CONTEXT + FRAME_SAMPLES],
            state: [0.0; STATE_SIZE],
        })
    }
}

impl SpeechDetector for SileroVad {
    fn probability(&mut self, frame: &[f32; FRAME_SAMPLES]) -> Result<f32> {
        self.input[CONTEXT..].copy_from_slice(frame);
        let outputs = self.session.run(ort::inputs! {
            "input" => TensorRef::from_array_view(([1, CONTEXT + FRAME_SAMPLES], &self.input[..]))?,
            "state" => TensorRef::from_array_view(([2, 1, 128], &self.state[..]))?,
            "sr" => TensorRef::from_array_view(([] as [usize; 0], &[16000_i64][..]))?,
        })?;
        let (_, probability) = outputs["output"].try_extract_tensor::<f32>()?;
        let (shape, state) = outputs["stateN"].try_extract_tensor::<f32>()?;
        if probability.len() != 1 || state.len() != STATE_SIZE || shape[..] != [2, 1, 128] {
            return Err("unexpected Silero VAD output dimensions".into());
        }
        let probability = probability[0];
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err("Silero VAD returned an invalid speech probability".into());
        }
        self.state.copy_from_slice(state);
        self.input.copy_within(FRAME_SAMPLES.., 0);
        Ok(probability)
    }

    fn reset(&mut self) {
        self.input.fill(0.0);
        self.state.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_not_speech_and_reset_is_reproducible() -> Result<()> {
        let mut vad = SileroVad::load(Path::new(DEFAULT_VAD_PATH))?;
        let silence = [0.0; FRAME_SAMPLES];
        let first = vad.probability(&silence)?;
        assert!(first < 0.1, "silence probability: {first}");
        for _ in 0..100 {
            assert!(vad.probability(&silence)? < 0.5);
        }
        vad.reset();
        assert!((vad.probability(&silence)? - first).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn steady_tone_does_not_trigger_speech() -> Result<()> {
        let mut vad = SileroVad::load(Path::new(DEFAULT_VAD_PATH))?;
        let mut frames_above_threshold = 0;
        for t in 0..100 {
            let frame = std::array::from_fn(|i| {
                (std::f32::consts::TAU * 440.0 * (t * FRAME_SAMPLES + i) as f32 / 16000.0).sin()
                    * 0.2
            });
            frames_above_threshold += usize::from(vad.probability(&frame)? >= 0.5);
        }
        assert!(
            frames_above_threshold < 3,
            "tone triggered {frames_above_threshold} frames"
        );
        Ok(())
    }

    #[test]
    fn probabilities_match_upstream_python_wrapper() -> Result<()> {
        // Independent ONNX Runtime/Python reference: zero state, 64-sample rolling
        // context, scalar int64 sample rate. Verifies streaming state/context feedback.
        let audio = crate::audio::read_wav(
            &Path::new(crate::model::DEFAULT_MODEL_DIR).join("test_wavs/0.wav"),
        )?;
        let mut vad = SileroVad::load(Path::new(DEFAULT_VAD_PATH))?;
        let expected = [
            (0, 0.014_339_328),
            (1, 0.020_782_024),
            (4, 0.007_357_657),
            (9, 0.010_054_916),
            (19, 0.997_940_36),
            (39, 0.998_451),
            (59, 0.999_924_66),
        ];
        for (t, frame) in audio
            .as_chunks::<FRAME_SAMPLES>()
            .0
            .iter()
            .take(60)
            .enumerate()
        {
            let actual = vad.probability(frame)?;
            if let Some((_, reference)) = expected.iter().find(|(index, _)| *index == t) {
                assert!(
                    (actual - reference).abs() < 1e-4,
                    "frame {t}: {actual} != {reference}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "timing measurement: cargo test --release vad::tests::benchmark -- --ignored --nocapture"]
    fn benchmark() -> Result<()> {
        let mut vad = SileroVad::load(Path::new(DEFAULT_VAD_PATH))?;
        let frame = [0.0; FRAME_SAMPLES];
        for _ in 0..50 {
            vad.probability(&frame)?;
        }
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            vad.probability(&frame)?;
        }
        let per_frame = start.elapsed().as_secs_f64() / 1000.0;
        eprintln!(
            "Silero VAD: {:.1} us / 32 ms frame, {:.2}% of one CPU core",
            per_frame * 1e6,
            per_frame / 0.032 * 100.0
        );
        Ok(())
    }
}
