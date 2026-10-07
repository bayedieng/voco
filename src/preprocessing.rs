//! Parakeet's normalized, channel-first 128-bin log-Mel frontend.
use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex32};

use crate::{Result, audio::SAMPLE_RATE};

const WINDOW: usize = 400;
const HOP: usize = 160;
const FFT_SIZE: usize = 512;
pub const MEL_BINS: usize = 128;

struct MelFilter {
    start: usize,
    weights: Vec<f32>,
}

pub struct Features {
    pub data: Vec<f32>,
    pub frames: usize,
}

/// Reuses FFT plans, scratch, periodic Hann window and sparse Slaney filters.
/// Settings match Sherpa's kaldi-native-fbank Parakeet reference frontend.
pub struct Preprocessor {
    fft: Arc<dyn RealToComplex<f32>>,
    window: [f32; WINDOW],
    filters: Vec<MelFilter>,
    input: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
    power: [f32; FFT_SIZE / 2 + 1],
}

impl Preprocessor {
    pub fn new() -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        let window = std::array::from_fn(|i| {
            (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / WINDOW as f64).cos()) as f32
        });
        let max_mel = 15.0 + 14.545_078_f32 * (8000.0_f32 / 1000.0).ln();
        let step = max_mel / (MEL_BINS + 1) as f32;
        let mel_to_hz = |mel: f32| {
            if mel <= 15.0 {
                200.0 / 3.0 * mel
            } else {
                1000.0 * ((mel - 15.0) * 0.068_751_775).exp()
            }
        };
        let filters = (0..MEL_BINS)
            .map(|m| {
                let left = mel_to_hz(m as f32 * step);
                let center = mel_to_hz((m + 1) as f32 * step);
                let right = mel_to_hz((m + 2) as f32 * step);
                let norm = 2.0 / (right - left);
                let mut start = 0;
                let mut weights = Vec::new();
                for k in 0..=FFT_SIZE / 2 {
                    let hz = k as f32 * SAMPLE_RATE as f32 / FFT_SIZE as f32;
                    if hz > left && hz < right {
                        if weights.is_empty() {
                            start = k;
                        }
                        let weight = if hz <= center {
                            (hz - left) / (center - left)
                        } else {
                            (right - hz) / (right - center)
                        };
                        weights.push(weight * norm);
                    }
                }
                assert!(!weights.is_empty());
                MelFilter { start, weights }
            })
            .collect();
        Self {
            window,
            filters,
            input: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
            power: [0.0; FFT_SIZE / 2 + 1],
        }
    }

    pub fn compute(&mut self, audio: &[f32]) -> Result<Features> {
        if audio.is_empty() || audio.iter().any(|x| !x.is_finite()) {
            return Err("audio is empty or contains non-finite samples".into());
        }
        // Virtual 2s reference tail; snip_edges=true. No padded waveform allocation.
        let frames = 1 + (audio.len() + 2 * SAMPLE_RATE - WINDOW) / HOP;
        let mut data = vec![0.0; MEL_BINS * frames];
        let silence = f32::EPSILON.ln();
        for t in 0..frames {
            let start = t * HOP;
            if start >= audio.len() {
                for m in 0..MEL_BINS {
                    data[m * frames + t] = silence;
                }
                continue;
            }
            self.input.fill(0.0);
            let count = WINDOW.min(audio.len() - start);
            let frame = &audio[start..start + count];
            // Frame-local pre-emphasis; no dither or DC-offset removal.
            for i in 0..WINDOW {
                let sample = frame.get(i).copied().unwrap_or(0.0);
                let previous = if i == 0 {
                    sample
                } else {
                    frame.get(i - 1).copied().unwrap_or(0.0)
                };
                self.input[i] = (sample - 0.97 * previous) * self.window[i];
            }
            self.fft.process_with_scratch(
                &mut self.input,
                &mut self.spectrum,
                &mut self.scratch,
            )?;
            for (power, complex) in self.power.iter_mut().zip(&self.spectrum) {
                *power = complex.norm_sqr();
            }
            for (m, filter) in self.filters.iter().enumerate() {
                let energy: f32 = filter
                    .weights
                    .iter()
                    .zip(&self.power[filter.start..filter.start + filter.weights.len()])
                    .map(|(w, p)| w * p)
                    .sum();
                // Write ONNX layout directly, avoiding a feature transpose/copy.
                data[m * frames + t] = energy.max(f32::EPSILON).ln();
            }
        }
        normalize(&mut data, frames);
        Ok(Features { data, frames })
    }
}

fn normalize(data: &mut [f32], frames: usize) {
    for channel in data.chunks_exact_mut(frames) {
        let mean = channel.iter().map(|&x| x as f64).sum::<f64>() / frames as f64;
        let variance = channel
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / (frames - 1).max(1) as f64;
        let scale = 1.0 / (variance.sqrt() + 1e-5);
        for x in channel {
            *x = ((*x as f64 - mean) * scale) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_has_expected_shape_and_normalizes_to_zero() {
        let features = Preprocessor::new()
            .compute(&vec![0.0; SAMPLE_RATE])
            .unwrap();
        assert_eq!(features.frames, 298);
        assert_eq!(features.data.len(), MEL_BINS * features.frames);
        assert!(features.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn features_are_finite_and_per_channel_normalized() {
        let audio: Vec<_> = (0..SAMPLE_RATE)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.5)
            .collect();
        let features = Preprocessor::new().compute(&audio).unwrap();
        for channel in features.data.chunks_exact(features.frames) {
            assert!(channel.iter().all(|x| x.is_finite()));
            let mean = channel.iter().sum::<f32>() / features.frames as f32;
            let variance =
                channel.iter().map(|x| x * x).sum::<f32>() / (features.frames - 1) as f32;
            assert!(mean.abs() < 1e-5);
            assert!((variance - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn features_match_kaldi_native_fbank_reference() {
        let audio: Vec<_> = (0..1600)
            .map(|i| ((i * 17) % 127 - 63) as f32 / 128.0)
            .collect();
        let features = Preprocessor::new().compute(&audio).unwrap();
        assert_eq!(features.frames, 208);
        for (t, m, expected) in [
            (0, 0, 3.504_366),
            (0, 32, 4.487_659),
            (3, 64, 4.457_354),
            (9, 127, 3.886_852_5),
            (10, 32, -0.224_117_82),
            (100, 64, -0.223_614_38),
        ] {
            let actual = features.data[m * features.frames + t];
            assert!(
                (actual - expected).abs() < 1e-3,
                "frame {t}, bin {m}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn normalization_uses_sample_standard_deviation() {
        let mut values = vec![1.0, 2.0, 3.0];
        normalize(&mut values, 3);
        assert!((values[0] + 1.0 / 1.00001).abs() < 1e-6);
        assert_eq!(values[1], 0.0);
    }

    #[test]
    fn invalid_audio_is_rejected() {
        let mut preprocessor = Preprocessor::new();
        assert!(preprocessor.compute(&[]).is_err());
        assert!(preprocessor.compute(&[f32::NAN]).is_err());
    }
}
