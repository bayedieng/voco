mod audio;
mod cli;
mod dictation;
mod microphone;
mod model;
mod output;
mod preprocessing;
mod utterance;
mod vad;

use std::{error::Error, path::Path, time::Instant};

use clap::Parser;

use crate::{audio::SAMPLE_RATE, cli::Args, model::Parakeet, preprocessing::Preprocessor};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn main() -> Result<()> {
    let args = Args::parse();
    if args.list_devices {
        microphone::list_devices()
    } else if args.mic {
        dictation::run(
            args.model_dir(),
            &args.vad_model,
            args.device.as_deref(),
            args.endpoint_config(),
            args.print_only,
        )
    } else {
        transcribe_wav(
            args.wav.as_deref().expect("CLI requires a mode"),
            args.model_dir(),
        )
    }
}

fn transcribe_wav(wav: &Path, model_dir: &Path) -> Result<()> {
    let start = Instant::now();
    let samples = audio::read_wav(wav)?;
    let duration = samples.len() as f64 / SAMPLE_RATE as f64;
    let features = Preprocessor::new().compute(&samples)?;
    let preprocessing = start.elapsed();
    let start = Instant::now();
    let mut model = Parakeet::load(model_dir)?;
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
