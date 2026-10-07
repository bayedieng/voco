//! Command-line modes and resource policy.
use crate::model::DEFAULT_MODEL_DIR;
use clap::{ArgGroup, Parser};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    version,
    about = "Whole-recording dictation with a left-hand-friendly toggle shortcut"
)]
#[command(group(ArgGroup::new("mode").required(true).args(["wav", "daemon", "toggle", "daemon_status", "quit", "preview_cues", "list_devices"])))]
pub struct Args {
    /// Transcribe a WAV file; does not inject text
    pub wav: Option<PathBuf>,
    #[arg(requires = "wav", conflicts_with = "model_dir")]
    pub legacy_model_dir: Option<PathBuf>,
    /// Run the toggle-to-record daemon in the current process (also accepts --mic)
    #[arg(long, alias = "mic")]
    pub daemon: bool,
    #[arg(long)]
    pub list_devices: bool,
    #[arg(long)]
    pub model_dir: Option<PathBuf>,
    /// Exact input device name; otherwise use the default device
    #[arg(long, requires = "daemon", conflicts_with_all = ["wav", "list_devices"])]
    pub device: Option<String>,
    /// Print dictation without typing into the focused application
    #[arg(long, requires = "daemon", conflicts_with_all = ["wav", "list_devices"])]
    pub print_only: bool,
    /// Toggle recording in the already running daemon
    #[arg(long)]
    pub toggle: bool,
    #[arg(long)]
    pub daemon_status: bool,
    /// Exit the running daemon (does not disable its service)
    #[arg(long)]
    pub quit: bool,
    /// Play recording-start and dictation-complete sounds and exit
    #[arg(long)]
    pub preview_cues: bool,
    /// Left-hand shortcut; Control+Option+Z on macOS
    #[arg(long, default_value = "Ctrl+Alt+Z")]
    pub hotkey: String,
    /// Let the desktop/compositor shortcut run vocod --toggle instead of registering one
    #[arg(long)]
    pub external_hotkey: bool,
    #[arg(long)]
    pub no_cues: bool,
    /// Private control socket; defaults to the per-user Voco data directory
    #[arg(long)]
    pub control_socket: Option<PathBuf>,
    /// Upper recording duration bound; the complete clip is transcribed on reaching it
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u64).range(1..=120))]
    pub max_recording_secs: u64,
    /// Free ASR sessions after this many idle seconds
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u64).range(1..=86400))]
    pub model_idle_secs: u64,
    /// Keep ASR loaded permanently to remove cold model-loading latency
    #[arg(long)]
    pub keep_model_loaded: bool,
}
impl Args {
    pub fn model_dir(&self) -> &Path {
        self.model_dir
            .as_deref()
            .or(self.legacy_model_dir.as_deref())
            .unwrap_or_else(|| Path::new(DEFAULT_MODEL_DIR))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn modes_are_exclusive_and_legacy_paths_work() {
        assert!(Args::try_parse_from(["vocod"]).is_err());
        assert!(Args::try_parse_from(["vocod", "--daemon", "test.wav"]).is_err());
        assert!(Args::try_parse_from(["vocod", "--print-only", "test.wav"]).is_err());
        let args = Args::try_parse_from(["vocod", "test.wav", "models"]).unwrap();
        assert_eq!(args.model_dir(), Path::new("models"));
        assert!(
            Args::try_parse_from(["vocod", "--mic", "--print-only"])
                .unwrap()
                .daemon
        );
        assert!(Args::try_parse_from(["vocod", "--model-idle-secs", "0", "--daemon"]).is_err());
    }
}
