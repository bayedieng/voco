//! Command-line configuration; audio/ONNX code does not depend on argument parsing.
use std::path::{Path, PathBuf};

use clap::{ArgGroup, Parser};

use crate::{model::DEFAULT_MODEL_DIR, utterance::EndpointConfig, vad::DEFAULT_VAD_PATH};

#[derive(Parser)]
#[command(version, about = "Parakeet WAV transcription and microphone dictation")]
#[command(group(ArgGroup::new("mode").required(true).args(["wav", "mic", "list_devices"])))]
pub struct Args {
    /// Transcribe a WAV file (does not type into other applications)
    pub wav: Option<PathBuf>,

    /// Legacy positional model directory for WAV mode
    #[arg(requires = "wav", conflicts_with = "model_dir")]
    pub legacy_model_dir: Option<PathBuf>,

    /// Continuously capture the microphone and type each completed utterance
    #[arg(long)]
    pub mic: bool,

    /// List available input devices and exit
    #[arg(long)]
    pub list_devices: bool,

    /// Directory containing encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt
    #[arg(long)]
    pub model_dir: Option<PathBuf>,

    /// Exact microphone name from --list-devices; otherwise use the default device
    #[arg(long, requires = "mic", conflicts_with_all = ["wav", "list_devices"])]
    pub device: Option<String>,

    /// Print mic transcriptions without typing (useful for tuning endpointing)
    #[arg(long, requires = "mic", conflicts_with_all = ["wav", "list_devices"])]
    pub print_only: bool,

    /// Silero VAD ONNX model (512-sample 16kHz streaming interface)
    #[arg(long, default_value = DEFAULT_VAD_PATH)]
    pub vad_model: PathBuf,

    /// Speech probability required to start an utterance (0..1); raise to reject more noise
    #[arg(long = "vad-threshold", alias = "threshold", default_value_t = 0.5)]
    pub threshold: f32,

    /// End an utterance after this much VAD-classified non-speech
    #[arg(long, default_value_t = 600)]
    pub silence_ms: u64,

    /// Reject bursts shorter than this much VAD-confirmed speech
    #[arg(long, default_value_t = 96)]
    pub min_speech_ms: u64,

    /// Split long utterances to bound inference latency and memory
    #[arg(long, default_value_t = 15)]
    pub max_utterance_secs: u64,
}

impl Args {
    pub fn model_dir(&self) -> &Path {
        self.model_dir
            .as_deref()
            .or(self.legacy_model_dir.as_deref())
            .unwrap_or_else(|| Path::new(DEFAULT_MODEL_DIR))
    }

    pub fn endpoint_config(&self) -> EndpointConfig {
        EndpointConfig {
            threshold: self.threshold,
            silence_ms: self.silence_ms,
            min_speech_ms: self.min_speech_ms,
            max_seconds: self.max_utterance_secs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_and_legacy_paths() {
        assert!(Args::try_parse_from(["vocod"]).is_err());
        assert!(Args::try_parse_from(["vocod", "--mic", "test.wav"]).is_err());
        assert!(Args::try_parse_from(["vocod", "--print-only", "test.wav"]).is_err());
        let args = Args::try_parse_from(["vocod", "test.wav", "models"]).unwrap();
        assert_eq!(args.model_dir(), Path::new("models"));
        let args = Args::try_parse_from(["vocod", "--mic", "--print-only"]).unwrap();
        assert!(args.mic && args.print_only);
        assert_eq!(
            args.endpoint_config().min_speech_ms,
            EndpointConfig::default().min_speech_ms
        );
    }
}
