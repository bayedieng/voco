//! WAV decoding and continuous, band-limited resampling to the model's sample rate.
use std::path::Path;

use rubato::{FftFixedInOut, Resampler};

use crate::Result;

pub const SAMPLE_RATE: usize = 16_000;

/// Select the first channel and scale integer PCM into normalized float samples.
pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        return Err("WAV must have a nonzero sample rate and channel count".into());
    }
    let channels = spec.channels as usize;
    let mut audio = Vec::with_capacity(reader.duration() as usize);
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for (i, sample) in reader.samples::<f32>().enumerate() {
                let sample = sample?;
                if i % channels == 0 {
                    audio.push(sample);
                }
            }
        }
        hound::SampleFormat::Int => {
            if !(1..=32).contains(&spec.bits_per_sample) {
                return Err("unsupported PCM bit depth".into());
            }
            let scale = 2.0_f32.powi(1 - spec.bits_per_sample as i32);
            for (i, sample) in reader.samples::<i32>().enumerate() {
                let sample = sample?;
                if i % channels == 0 {
                    audio.push(sample as f32 * scale);
                }
            }
        }
    }
    if audio.is_empty() || audio.iter().any(|x| !x.is_finite()) {
        return Err("WAV is empty or contains non-finite samples".into());
    }
    if spec.sample_rate as usize == SAMPLE_RATE {
        return Ok(audio);
    }
    let mut resampler = StreamingResampler::new(spec.sample_rate as usize)?;
    let mut result = Vec::with_capacity(audio.len() * SAMPLE_RATE / spec.sample_rate as usize + 1);
    resampler.push(&audio, &mut result)?;
    resampler.finish(&mut result)?;
    Ok(result)
}

/// Keeps filter history across capture chunks; no per-chunk allocation or resampler rebuild.
pub struct StreamingResampler {
    fft: Option<FftFixedInOut<f32>>,
    rate: usize,
    input: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
    filled: usize,
    discard: usize,
    input_count: usize,
    output_count: usize,
}

impl StreamingResampler {
    pub fn new(rate: usize) -> Result<Self> {
        if rate == 0 {
            return Err("sample rate must be positive".into());
        }
        let fft = if rate == SAMPLE_RATE {
            None
        } else {
            Some(FftFixedInOut::<f32>::new(rate, SAMPLE_RATE, 1024, 1)?)
        };
        let input = fft
            .as_ref()
            .map(|r| vec![vec![0.0; r.input_frames_next()]])
            .unwrap_or_default();
        let output = fft
            .as_ref()
            .map(|r| r.output_buffer_allocate(true))
            .unwrap_or_default();
        let discard = fft.as_ref().map_or(0, Resampler::output_delay);
        Ok(Self {
            fft,
            rate,
            input,
            output,
            discard,
            filled: 0,
            input_count: 0,
            output_count: 0,
        })
    }

    pub fn push(&mut self, mut samples: &[f32], destination: &mut Vec<f32>) -> Result<()> {
        self.input_count += samples.len();
        if self.fft.is_none() {
            destination.extend_from_slice(samples);
            self.output_count += samples.len();
            return Ok(());
        }
        while !samples.is_empty() {
            let n = samples.len().min(self.input[0].len() - self.filled);
            self.input[0][self.filled..self.filled + n].copy_from_slice(&samples[..n]);
            self.filled += n;
            samples = &samples[n..];
            if self.filled == self.input[0].len() {
                self.process(destination, usize::MAX)?;
            }
        }
        Ok(())
    }

    /// Flush only when capture ends, not between utterances (which share filter history).
    pub fn finish(&mut self, destination: &mut Vec<f32>) -> Result<()> {
        if self.fft.is_none() {
            return Ok(());
        }
        let expected =
            (self.input_count as u64 * SAMPLE_RATE as u64).div_ceil(self.rate as u64) as usize;
        while self.output_count < expected {
            self.input[0][self.filled..].fill(0.0);
            self.process(destination, expected - self.output_count)?;
        }
        Ok(())
    }

    fn process(&mut self, destination: &mut Vec<f32>, limit: usize) -> Result<()> {
        let (_, written) = self
            .fft
            .as_mut()
            .expect("resampling enabled")
            .process_into_buffer(&self.input, &mut self.output, None)?;
        self.filled = 0;
        let skip = self.discard.min(written);
        self.discard -= skip;
        let n = (written - skip).min(limit);
        destination.extend_from_slice(&self.output[0][skip..skip + n]);
        self.output_count += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_preserves_duration_and_passband_amplitude() {
        for rate in [8000, 44100, 48000] {
            let input: Vec<_> = (0..rate)
                .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / rate as f32).sin() * 0.5)
                .collect();
            let mut resampler = StreamingResampler::new(rate).unwrap();
            let mut output = Vec::new();
            for chunk in input.chunks(137) {
                resampler.push(chunk, &mut output).unwrap();
            }
            resampler.finish(&mut output).unwrap();
            assert_eq!(output.len(), SAMPLE_RATE);
            let rms = (output[1000..15000].iter().map(|x| x * x).sum::<f32>() / 14000.0).sqrt();
            assert!((rms - 0.5 / 2.0_f32.sqrt()).abs() < 0.01);
        }
    }

    #[test]
    fn chunk_boundaries_do_not_change_resampling() {
        let audio: Vec<_> = (0..5000).map(|i| (i as f32 * 0.1).sin()).collect();
        let convert = |chunk_size| {
            let mut resampler = StreamingResampler::new(48000).unwrap();
            let mut output = Vec::new();
            for chunk in audio.chunks(chunk_size) {
                resampler.push(chunk, &mut output).unwrap();
            }
            resampler.finish(&mut output).unwrap();
            output
        };
        assert_eq!(convert(17), convert(audio.len()));
    }

    #[test]
    fn passthrough_and_short_input() {
        let mut direct = StreamingResampler::new(SAMPLE_RATE).unwrap();
        let mut output = Vec::new();
        direct.push(&[0.1, 0.2], &mut output).unwrap();
        direct.finish(&mut output).unwrap();
        assert_eq!(output, [0.1, 0.2]);
        let mut resampler = StreamingResampler::new(48000).unwrap();
        output.clear();
        resampler.push(&[0.5; 10], &mut output).unwrap();
        resampler.finish(&mut output).unwrap();
        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|x| x.is_finite()));
    }
}
