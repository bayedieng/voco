//! Offline greedy TDT decoding. Sessions are loaded once and reused per utterance.
use std::path::Path;

use ort::{
    session::Session,
    value::{DynValue, Tensor, TensorRef},
};

use crate::{
    Result,
    preprocessing::{Features, MEL_BINS},
};

pub const DEFAULT_MODEL_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
);
const ENCODER_DIM: usize = 1024;
const HIDDEN: usize = 640;
const LAYERS: usize = 2;
const BLANK: usize = 1024;
const VOCAB_SIZE: usize = BLANK + 1;
const DURATIONS: usize = 5;

struct DecoderOutput {
    features: DynValue,
    hidden: DynValue,
    cell: DynValue,
}

pub struct Parakeet {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    tokens: Vec<String>,
}

impl Parakeet {
    pub fn load(dir: &Path) -> Result<Self> {
        let threads = std::thread::available_parallelism()?.get().min(4);
        let session = |name: &str, threads: usize| -> Result<Session> {
            Ok(Session::builder()?
                .with_intra_threads(threads)
                .map_err(ort::Error::<()>::from)?
                .with_inter_threads(1)
                .map_err(ort::Error::<()>::from)?
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

    pub fn transcribe(&mut self, features: Features) -> Result<String> {
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
                prediction = self.decode(token, &prediction.hidden, &prediction.cell)?;
                symbols_at_frame += 1;
            }
            // Duration classes are 0..4 frames. Blank/zero must still make progress.
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
    use crate::{audio::read_wav, preprocessing::Preprocessor};

    #[test]
    #[ignore = "loads large ONNX models; cargo test --release -- --ignored"]
    fn bundled_wav_transcribes() -> Result<()> {
        let dir = Path::new(DEFAULT_MODEL_DIR);
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
}
