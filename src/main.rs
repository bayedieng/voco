mod audio;
mod cli;
#[cfg(unix)]
mod control;
#[cfg(unix)]
mod hotkey;
#[cfg(unix)]
mod manual;
mod microphone;
mod model;
mod model_cache;
mod output;
mod preprocessing;
mod sound;
mod wake;

use std::{error::Error, path::Path, time::Instant};

use clap::Parser;

use crate::{audio::SAMPLE_RATE, cli::Args, model::Parakeet, preprocessing::Preprocessor};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn main() -> Result<()> {
    let args = Args::parse();
    if args.preview_cues {
        return sound::preview();
    }
    if args.daemon || args.toggle || args.daemon_status || args.quit {
        return daemon_command(&args);
    }
    if args.list_devices {
        microphone::list_devices()
    } else {
        transcribe_wav(
            args.wav.as_deref().expect("CLI requires a mode"),
            args.model_dir(),
        )
    }
}

fn daemon_command(args: &Args) -> Result<()> {
    #[cfg(unix)]
    {
        let socket = args
            .control_socket
            .clone()
            .map(Ok)
            .unwrap_or_else(control::default_socket)?;
        if !args.daemon {
            let action = if args.toggle {
                "toggle"
            } else if args.quit {
                "quit"
            } else {
                "status"
            };
            print!("{}", control::request(&socket, action)?);
            return Ok(());
        }
        manual::run(manual::Options {
            model_dir: args.model_dir().to_owned(),
            device: args.device.clone(),
            socket,
            shortcut: args.hotkey.clone(),
            external_hotkey: args.external_hotkey,
            no_cues: args.no_cues,
            print_only: args.print_only,
            idle_timeout: std::time::Duration::from_secs(args.model_idle_secs),
            keep_loaded: args.keep_model_loaded,
            max_recording: std::time::Duration::from_secs(args.max_recording_secs),
        })
    }
    #[cfg(not(unix))]
    {
        let _ = args;
        Err("daemon control currently supports Linux and macOS".into())
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
