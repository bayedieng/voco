use std::{error::Error, path::Path, sync::Arc, time::Instant};

use ort::{
    session::Session,
    value::{DynValue, Tensor, TensorRef},
};
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex32};
use rubato::{FftFixedInOut, Resampler};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MODEL_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
);
const SAMPLE_RATE: usize = 16_000;
const WINDOW: usize = 400;
const HOP: usize = 160;
const FFT_SIZE: usize = 512;
const MEL_BINS: usize = 128;
const ENCODER_DIM: usize = 1024;
const HIDDEN: usize = 640;
const LAYERS: usize = 2;
const BLANK: usize = 1024;
const VOCAB_SIZE: usize = BLANK + 1;
const DURATIONS: usize = 5; // TDT duration classes represent 0, 1, 2, 3, 4 frames.

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let Some(wav) = args.next() else {
        eprintln!("Usage: vocod <audio.wav> [model-directory]");
        std::process::exit(2);
    };
    if wav == "--help" || wav == "-h" {
        println!("Usage: vocod <audio.wav> [model-directory]");
        return Ok(());
    }
    let model_dir = args.next().unwrap_or_else(|| MODEL_DIR.into());
    if args.next().is_some() {
        return Err("too many arguments: expected <audio.wav> [model-directory]".into());
    }

    let start = Instant::now();
    let samples = read_wav(Path::new(&wav))?;
    let duration = samples.len() as f64 / SAMPLE_RATE as f64;
    let mut preprocessor = Preprocessor::new();
    let features = preprocessor.compute(&samples)?;
    let preprocessing = start.elapsed();

    let start = Instant::now();
    let mut model = Parakeet::load(Path::new(&model_dir))?;
    let loading = start.elapsed();
    let start = Instant::now();
    let text = model.transcribe(features)?;
    let inference = start.elapsed();
    println!("{text}");
    eprintln!(
        "Audio: {duration:.2}s | preprocessing: {:.1}ms | model loading: {:.2}s | inference: {:.2}s | RTF: {:.3}",
        preprocessing.as_secs_f64() * 1000.0,
        loading.as_secs_f64(),
        inference.as_secs_f64(),
        (preprocessing + inference).as_secs_f64() / duration,
    );
    Ok(())
}

/// Read normalized PCM/float WAV, selecting the first channel like Sherpa's reference.
fn read_wav(path: &Path) -> Result<Vec<f32>> {
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
    resample(audio, spec.sample_rate as usize)
}

/// Band-limited FFT resampling, with reusable input/output buffers and delay compensation.
fn resample(audio: Vec<f32>, rate: usize) -> Result<Vec<f32>> {
    if rate == SAMPLE_RATE {
        return Ok(audio);
    }
    let mut resampler = FftFixedInOut::<f32>::new(rate, SAMPLE_RATE, 1024, 1)?;
    let chunk = resampler.input_frames_next();
    let delay = resampler.output_delay();
    let length = (audio.len() as u64 * SAMPLE_RATE as u64).div_ceil(rate as u64) as usize;
    let mut input = vec![vec![0.0; chunk]];
    let mut output = resampler.output_buffer_allocate(true);
    let mut result = Vec::with_capacity(length);
    let mut position = 0;
    let mut discard = delay;
    while result.len() < length {
        input[0].fill(0.0);
        let count = chunk.min(audio.len().saturating_sub(position));
        input[0][..count].copy_from_slice(&audio[position..position + count]);
        position += count;
        let (_, written) = resampler.process_into_buffer(&input, &mut output, None)?;
        let skip = discard.min(written);
        discard -= skip;
        let count = (written - skip).min(length - result.len());
        result.extend_from_slice(&output[0][skip..skip + count]);
    }
    Ok(result)
}

struct MelFilter {
    start: usize,
    weights: Vec<f32>,
}

struct Features {
    // Contiguous channel-first layout, ready for ONNX without a transpose/copy.
    data: Vec<f32>,
    frames: usize,
}

/// Matches kaldi-native-fbank's NeMo/Librosa settings used by Sherpa's Parakeet script.
/// FFT plans, periodic Hann window, sparse Slaney filters, and scratch are reused.
struct Preprocessor {
    fft: Arc<dyn RealToComplex<f32>>,
    window: [f32; WINDOW],
    filters: Vec<MelFilter>,
    input: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
    power: [f32; FFT_SIZE / 2 + 1],
}

impl Preprocessor {
    fn new() -> Self {
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

    fn compute(&mut self, audio: &[f32]) -> Result<Features> {
        if audio.is_empty() || audio.iter().any(|x| !x.is_finite()) {
            return Err("audio is empty or contains non-finite samples".into());
        }
        // Match the reference's two-second tail, without allocating a padded waveform.
        // snip_edges=true: only complete 400-sample frames are extracted.
        let frames = 1 + (audio.len() + 2 * SAMPLE_RATE - WINDOW) / HOP;
        let mut data = vec![0.0; MEL_BINS * frames];
        let silence = f32::EPSILON.ln();
        for t in 0..frames {
            let start = t * HOP;
            if start >= audio.len() {
                for m in 0..MEL_BINS {
                    data[m * frames + t] = silence;
                }
                continue; // Skip FFT and Mel evaluation for wholly silent tail frames.
            }
            self.input.fill(0.0);
            let count = WINDOW.min(audio.len() - start);
            let frame = &audio[start..start + count];
            // Pre-emphasis is frame-local, not carried over from the previous frame.
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
                data[m * frames + t] = energy.max(f32::EPSILON).ln();
            }
        }
        normalize(&mut data, frames);
        Ok(Features { data, frames })
    }
}

fn normalize(data: &mut [f32], frames: usize) {
    for channel in data.chunks_exact_mut(frames) {
        // Double-precision accumulation avoids cancellation on long/silent utterances.
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

struct DecoderOutput {
    features: DynValue,
    hidden: DynValue,
    cell: DynValue,
}

struct Parakeet {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    tokens: Vec<String>,
}

impl Parakeet {
    fn load(dir: &Path) -> Result<Self> {
        let threads = std::thread::available_parallelism()?.get().min(4);
        let session = |name: &str, threads: usize| -> Result<Session> {
            Ok(Session::builder()?
                .with_intra_threads(threads)?
                .with_inter_threads(1)?
                .commit_from_file(dir.join(name))?)
        };
        let mut tokens = vec![String::new(); VOCAB_SIZE];
        for line in std::fs::read_to_string(dir.join("tokens.txt"))?.lines() {
            let (token, id) = line.rsplit_once(' ').ok_or("invalid tokens.txt line")?;
            let id: usize = id.parse()?;
            if id >= tokens.len() || !tokens[id].is_empty() {
                return Err("unexpected or duplicate token ID".into());
            }
            tokens[id] = token.to_owned();
        }
        if tokens.iter().any(String::is_empty) || tokens[BLANK] != "<blk>" {
            return Err("expected Parakeet vocabulary with blank token 1024".into());
        }
        Ok(Self {
            encoder: session("encoder.int8.onnx", threads)?,
            decoder: session("decoder.int8.onnx", 1)?,
            joiner: session("joiner.int8.onnx", 1)?,
            tokens,
        })
    }

    fn decode(
        &mut self,
        token: usize,
        hidden: &DynValue,
        cell: &DynValue,
    ) -> Result<DecoderOutput> {
        let mut outputs = self.decoder.run(ort::inputs! {
            "targets" => TensorRef::from_array_view(([1, 1], &[token as i32][..]))?,
            "target_length" => TensorRef::from_array_view(([1], &[1_i32][..]))?,
            "states.1" => hidden,
            "onnx::Slice_3" => cell,
        })?;
        Ok(DecoderOutput {
            features: outputs
                .remove("outputs")
                .ok_or("missing decoder features")?,
            hidden: outputs
                .remove("states")
                .ok_or("missing decoder hidden state")?,
            cell: outputs.remove("162").ok_or("missing decoder cell state")?,
        })
    }

    fn transcribe(&mut self, features: Features) -> Result<String> {
        // Detach the output values before borrowing the other sessions.
        let (encoded, valid_frames) = {
            let mut outputs = self.encoder.run(ort::inputs! {
                "audio_signal" => Tensor::from_array(([1, MEL_BINS, features.frames], features.data))?,
                "length" => TensorRef::from_array_view(([1], &[features.frames as i64][..]))?,
            })?;
            let lengths = outputs["encoded_lengths"].try_extract_tensor::<i64>()?.1;
            let valid_frames = usize::try_from(*lengths.first().ok_or("missing encoded length")?)?;
            (
                outputs.remove("outputs").ok_or("missing encoder output")?,
                valid_frames,
            )
        };
        let (shape, data) = encoded.try_extract_tensor::<f32>()?;
        if shape.len() != 3 || shape[0] != 1 || shape[1] != ENCODER_DIM as i64 {
            return Err(format!("unexpected encoder shape: {shape:?}").into());
        }
        let stride = usize::try_from(shape[2])?;
        if valid_frames > stride {
            return Err("encoded length exceeds encoder output".into());
        }
        let hidden =
            Tensor::from_array(([LAYERS, 1, HIDDEN], vec![0_f32; LAYERS * HIDDEN]))?.into_dyn();
        let cell =
            Tensor::from_array(([LAYERS, 1, HIDDEN], vec![0_f32; LAYERS * HIDDEN]))?.into_dyn();
        let mut prediction = self.decode(BLANK, &hidden, &cell)?;
        let mut frame = [0_f32; ENCODER_DIM];
        let mut text = String::new();
        let mut t = 0;
        let mut symbols_at_frame = 0;
        while t < valid_frames {
            for (channel, x) in frame.iter_mut().enumerate() {
                *x = data[channel * stride + t];
            }
            let (token, duration) = {
                let outputs = self.joiner.run(ort::inputs! {
                    "encoder_outputs" => TensorRef::from_array_view(([1, ENCODER_DIM, 1], &frame[..]))?,
                    "decoder_outputs" => &prediction.features,
                })?;
                let logits = outputs["outputs"].try_extract_tensor::<f32>()?.1;
                if logits.len() != VOCAB_SIZE + DURATIONS {
                    return Err("unexpected joiner vocabulary/duration dimensions".into());
                }
                (argmax(&logits[..VOCAB_SIZE]), argmax(&logits[VOCAB_SIZE..]))
            };
            if token != BLANK {
                text.push_str(&self.tokens[token]);
                // Blank does not advance recurrent state; only emitted tokens do.
                prediction = self.decode(token, &prediction.hidden, &prediction.cell)?;
                symbols_at_frame += 1;
            }
            // Zero-duration nonblank emissions stay on the same encoder frame.
            // Force progress for blank/zero or pathological repeated emissions.
            let advance = if duration == 0 && (token == BLANK || symbols_at_frame >= 10) {
                1
            } else {
                duration
            };
            if advance > 0 {
                t += advance;
                symbols_at_frame = 0;
            }
        }
        Ok(text.replace('▁', " ").trim().to_owned())
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
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
        // Golden values from kaldi-native-fbank with the settings above, 2s tail,
        // and per-feature sample standard deviation. Covers partial/tail frames.
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
    #[ignore = "loads the large ONNX models; run with cargo test --release -- --ignored"]
    fn bundled_wav_transcribes() -> Result<()> {
        let dir = Path::new(MODEL_DIR);
        let audio = read_wav(&dir.join("test_wavs/0.wav"))?;
        let features = Preprocessor::new().compute(&audio)?;
        let text = Parakeet::load(dir)?.transcribe(features)?;
        assert!(
            text.starts_with("Well, I don't wish to see it any more"),
            "{text}"
        );
        assert!(text.ends_with("the old portrait."), "{text}");
        Ok(())
    }

    #[test]
    fn normalization_uses_sample_standard_deviation() {
        let mut values = vec![1.0, 2.0, 3.0];
        normalize(&mut values, 3);
        assert!((values[0] + 1.0 / 1.00001).abs() < 1e-6);
        assert_eq!(values[1], 0.0);
    }

    #[test]
    fn resampling_preserves_duration_and_passband_amplitude() {
        for rate in [8000, 44100, 48000] {
            let input = (0..rate)
                .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / rate as f32).sin() * 0.5)
                .collect();
            let output = resample(input, rate).unwrap();
            assert_eq!(output.len(), SAMPLE_RATE);
            let rms = (output[1000..15000].iter().map(|x| x * x).sum::<f32>() / 14000.0).sqrt();
            assert!((rms - 0.5 / 2.0_f32.sqrt()).abs() < 0.01);
        }
    }

    #[test]
    fn invalid_audio_is_rejected() {
        let mut preprocessor = Preprocessor::new();
        assert!(preprocessor.compute(&[]).is_err());
        assert!(preprocessor.compute(&[f32::NAN]).is_err());
    }
}
