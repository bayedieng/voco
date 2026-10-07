//! Short CC0 Kenney cues. Output devices exist only while a cue is playing.
use crate::Result;
use cpal::{
    FromSample, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::{
    io::Cursor,
    sync::mpsc::{self, SyncSender},
    thread::{self, JoinHandle},
    time::Duration,
};

#[derive(Clone, Copy)]
pub enum Cue {
    Start,
    Done,
}
#[derive(Clone, Default)]
pub struct Cues {
    sender: Option<SyncSender<Cue>>,
}
impl Cues {
    pub fn play(&self, cue: Cue) {
        if let Some(sender) = &self.sender {
            let _ = sender.try_send(cue);
        }
    }
}

pub struct Player {
    pub cues: Cues,
    worker: Option<JoinHandle<()>>,
}
impl Player {
    pub fn new(disabled: bool) -> Result<Self> {
        if disabled {
            return Ok(Self {
                cues: Cues::default(),
                worker: None,
            });
        }
        let (sender, receiver) = mpsc::sync_channel(4);
        let worker = thread::Builder::new()
            .name("audio-cues".into())
            .spawn(move || {
                let mut enabled = true;
                while let Ok(cue) = receiver.recv() {
                    if enabled && let Err(error) = play(cue) {
                        eprintln!("Audio cues disabled: {error}");
                        enabled = false;
                    }
                }
            })?;
        Ok(Self {
            cues: Cues {
                sender: Some(sender),
            },
            worker: Some(worker),
        })
    }
    pub fn finish(mut self) {
        self.cues.sender = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn decode(cue: Cue) -> Result<(Vec<f32>, u32)> {
    let bytes: &[u8] = match cue {
        Cue::Start => include_bytes!(concat!(env!("OUT_DIR"), "/cue-start.wav")),
        Cue::Done => include_bytes!(concat!(env!("OUT_DIR"), "/cue-done.wav")),
    };
    let mut reader = hound::WavReader::new(Cursor::new(bytes))?;
    let spec = reader.spec();
    let pcm = reader
        .samples::<i16>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let samples = pcm
        .chunks_exact(spec.channels as usize)
        .map(|frame| {
            frame.iter().map(|&x| x as f32 / 32768.0).sum::<f32>() / spec.channels as f32 * 0.4
        })
        .collect();
    Ok((samples, spec.sample_rate))
}

fn play(cue: Cue) -> Result<()> {
    let (samples, rate) = decode(cue)?;
    let device = cpal::default_host()
        .default_output_device()
        .ok_or("no audio output device")?;
    let supported = device.default_output_config()?;
    let config: cpal::StreamConfig = supported.clone().into();
    let length =
        (samples.len() as u64 * config.sample_rate.0 as u64).div_ceil(rate as u64) as usize;
    let audio: Vec<f32> = (0..length)
        .map(|i| {
            let position = i as f64 * rate as f64 / config.sample_rate.0 as f64;
            let index = position as usize;
            let a = samples[index.min(samples.len() - 1)];
            let b = samples[(index + 1).min(samples.len() - 1)];
            a + (b - a) * (position - index as f64) as f32
        })
        .collect();
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => output::<f32>(&device, &config, audio)?,
        cpal::SampleFormat::I16 => output::<i16>(&device, &config, audio)?,
        cpal::SampleFormat::U16 => output::<u16>(&device, &config, audio)?,
        format => return Err(format!("unsupported output format: {format}").into()),
    };
    stream.play()?;
    // A single timed wait, not a callback/polling loop; close the device after playback.
    thread::sleep(Duration::from_secs_f64(
        length as f64 / config.sample_rate.0 as f64 + 0.1,
    ));
    drop(stream);
    Ok(())
}

fn output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    audio: Vec<f32>,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let mut position = 0;
    Ok(device.build_output_stream(
        config,
        move |out: &mut [T], _| {
            for frame in out.chunks_exact_mut(channels) {
                let sample = audio.get(position).copied().unwrap_or(0.0);
                position += 1;
                frame.fill(T::from_sample(sample));
            }
        },
        |_| {},
        None,
    )?)
}

pub fn preview() -> Result<()> {
    play(Cue::Start)?;
    thread::sleep(Duration::from_millis(250));
    play(Cue::Done)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn embedded_cues_are_short_and_valid() -> Result<()> {
        for cue in [Cue::Start, Cue::Done] {
            let (samples, rate) = decode(cue)?;
            assert!(!samples.is_empty());
            assert!(samples.len() as f64 / (rate as f64) < 0.1);
            assert!(samples.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
        }
        Ok(())
    }
}
