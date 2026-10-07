//! Real-time CPAL capture. The callback only converts samples and writes to an SPSC ring.
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use cpal::{
    FromSample, Sample, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::{Result, wake::Wake};

#[derive(Clone)]
struct CallbackControl {
    stop: Arc<AtomicBool>,
    dropped: Arc<AtomicUsize>,
    failed: Arc<AtomicBool>,
    wake: Arc<Wake>,
}

pub struct Capture {
    stream: cpal::Stream,
    pub samples: Consumer<f32>,
    pub sample_rate: usize,
    pub dropped: Arc<AtomicUsize>,
    pub failed: Arc<AtomicBool>,
}

pub fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    let default = host.default_input_device().and_then(|d| d.name().ok());
    for device in host.input_devices()? {
        let name = device.name()?;
        let suffix = if default.as_deref() == Some(&name) {
            " (default)"
        } else {
            ""
        };
        println!("{name}{suffix}");
    }
    Ok(())
}

impl Capture {
    pub fn open_quiet(
        name: Option<&str>,
        stop: Arc<AtomicBool>,
        wake: Arc<Wake>,
        quiet: bool,
    ) -> Result<Self> {
        let host = cpal::default_host();
        let device = if let Some(name) = name {
            host.input_devices()?
                .find(|d| d.name().is_ok_and(|n| n == name))
                .ok_or_else(|| format!("input device not found: {name}"))?
        } else {
            host.default_input_device()
                .ok_or("no default microphone available")?
        };
        // Avoid our resampler when the host can directly supply mono 16kHz.
        let supported = device
            .supported_input_configs()
            .ok()
            .and_then(|mut configs| {
                configs
                    .find(|c| {
                        c.channels() == 1
                            && c.min_sample_rate().0 <= 16000
                            && c.max_sample_rate().0 >= 16000
                            && matches!(
                                c.sample_format(),
                                cpal::SampleFormat::F32 | cpal::SampleFormat::I16
                            )
                    })
                    .map(|c| c.with_sample_rate(cpal::SampleRate(16000)))
            })
            .map(Ok)
            .unwrap_or_else(|| device.default_input_config())?;
        let config: cpal::StreamConfig = supported.clone().into();
        let channels = config.channels as usize;
        let sample_rate = config.sample_rate.0 as usize;
        if channels == 0 || sample_rate == 0 {
            return Err("invalid microphone configuration".into());
        }
        // Two seconds of mono native-rate audio. Bound memory and do not block on overflow.
        let (producer, samples) = RingBuffer::new(sample_rate * 2);
        let dropped = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicBool::new(false));
        let control = CallbackControl {
            stop,
            dropped: Arc::clone(&dropped),
            failed: Arc::clone(&failed),
            wake,
        };
        macro_rules! build {
            ($ty:ty) => {
                build_stream::<$ty>(&device, &config, channels, producer, control)?
            };
        }
        let stream = match supported.sample_format() {
            cpal::SampleFormat::I8 => build!(i8),
            cpal::SampleFormat::I16 => build!(i16),
            cpal::SampleFormat::I32 => build!(i32),
            cpal::SampleFormat::I64 => build!(i64),
            cpal::SampleFormat::U8 => build!(u8),
            cpal::SampleFormat::U16 => build!(u16),
            cpal::SampleFormat::U32 => build!(u32),
            cpal::SampleFormat::U64 => build!(u64),
            cpal::SampleFormat::F32 => build!(f32),
            cpal::SampleFormat::F64 => build!(f64),
            format => return Err(format!("unsupported microphone sample format: {format}").into()),
        };
        if !quiet {
            eprintln!(
                "Microphone: {} | {sample_rate} Hz, {channels} channel(s)",
                device.name()?
            );
        }
        Ok(Self {
            stream,
            samples,
            sample_rate,
            dropped,
            failed,
        })
    }

    pub fn start(&self) -> Result<()> {
        self.stream.play()?;
        Ok(())
    }

    /// Keep the stream alive on its owning thread while the consumer moves to a worker.
    pub fn into_parts(
        self,
    ) -> (
        cpal::Stream,
        Consumer<f32>,
        usize,
        Arc<AtomicUsize>,
        Arc<AtomicBool>,
    ) {
        (
            self.stream,
            self.samples,
            self.sample_rate,
            self.dropped,
            self.failed,
        )
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    mut producer: Producer<f32>,
    control: CallbackControl,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let error_control = control.clone();
    let capacity = config.sample_rate.0 as usize * 2;
    let notify_at = config.sample_rate.0 as usize * 32 / 1000;
    Ok(device.build_input_stream(
        config,
        move |data: &[T], _| {
            if control.stop.load(Ordering::Relaxed) {
                return;
            }
            let mut lost = 0;
            for frame in data.chunks_exact(channels) {
                let sample = f32::from_sample(frame[0]);
                let sample = if sample.is_finite() {
                    sample.clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                if producer.push(sample).is_err() {
                    lost += 1;
                }
            }
            if lost != 0 {
                control.dropped.fetch_add(lost, Ordering::Relaxed);
            }
            // Batch wakeups at ~32 ms instead of waking for every tiny callback.
            if lost != 0 || capacity - producer.slots() >= notify_at {
                control.wake.notify();
            }
        },
        move |_| {
            error_control.failed.store(true, Ordering::Relaxed);
            error_control.wake.notify();
        },
        None,
    )?)
}
