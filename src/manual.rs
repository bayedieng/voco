//! Toggle-to-record daemon: closed microphone at idle, whole-recording ASR, model prefetch.
use crate::{
    Result,
    audio::{SAMPLE_RATE, StreamingResampler},
    control::{self, Control, IDLE, RECORDING, TRANSCRIBING},
    hotkey,
    microphone::Capture,
    model::Parakeet,
    model_cache::ModelCache,
    output::DictationOutput,
    preprocessing::Preprocessor,
    sound::{Cue, Cues, Player},
    wake::Wake,
};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

pub struct Options {
    pub model_dir: PathBuf,
    pub device: Option<String>,
    pub socket: PathBuf,
    pub shortcut: String,
    pub external_hotkey: bool,
    pub no_cues: bool,
    pub print_only: bool,
    pub idle_timeout: Duration,
    pub keep_loaded: bool,
    pub max_recording: Duration,
}

struct StopOnDrop(Arc<Control>);
impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.shutdown();
    }
}

pub fn run(options: Options) -> Result<()> {
    unsafe extern "C" {
        fn geteuid() -> u32;
    }
    // SAFETY: get effective Unix UID; this never modifies credentials or requests elevation.
    if unsafe { geteuid() } == 0 {
        return Err(
            "Run the dictation daemon as your logged-in desktop user, not root/sudo".into(),
        );
    }
    let (commands, receiver) = mpsc::sync_channel(16);
    let control = Arc::new(Control {
        stop: AtomicBool::new(false),
        state: AtomicU8::new(IDLE),
        keys_down: AtomicBool::new(false),
        warmup: AtomicBool::new(false),
        recorder: Arc::new(Wake::default()),
        inference: Arc::new(Wake::default()),
        ui: Arc::new(Wake::default()),
        toggles: commands,
    });
    control.ui.register();
    let _stop = StopOnDrop(Arc::clone(&control));
    let (listener, _socket_guard) = control::bind(&options.socket)?;
    let signal = Arc::clone(&control);
    ctrlc::set_handler(move || signal.shutdown())?;
    let _hotkey = hotkey::register(
        &options.shortcut,
        options.external_hotkey,
        Arc::clone(&control),
    )?;
    let sounds = Player::new(options.no_cues)?;
    let socket_path = options.socket.clone();
    let socket_control = Arc::clone(&control);
    let socket_worker = thread::Builder::new()
        .name("daemon-control".into())
        .spawn(move || {
            let result = control::serve(listener, Arc::clone(&socket_control));
            if result.is_err() {
                socket_control.shutdown();
            }
            result
        })?;
    let (audio_tx, audio_rx) = mpsc::sync_channel(1);
    let inference_control = Arc::clone(&control);
    let inference_cues = sounds.cues.clone();
    let model_dir = options.model_dir.clone();
    let timeout = if options.keep_loaded {
        None
    } else {
        Some(options.idle_timeout)
    };
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let processor = thread::Builder::new()
        .name("speech-inference".into())
        .spawn(move || {
            let _stop = StopOnDrop(Arc::clone(&inference_control));
            let result = inference(
                audio_rx,
                model_dir,
                timeout,
                options.keep_loaded,
                options.print_only,
                inference_cues,
                Arc::clone(&inference_control),
                ready_tx,
            );
            inference_control.shutdown();
            result
        })?;
    // Check desktop permissions early, rather than silently failing after the first recording.
    if !ready_rx.recv()? {
        control.shutdown();
        let _ = control::request(&socket_path, "quit");
        let processed = processor.join().map_err(|_| "inference worker panicked")?;
        let _ = socket_worker.join();
        sounds.finish();
        return processed;
    }
    let recorder_control = Arc::clone(&control);
    let recorder_cues = sounds.cues.clone();
    let recorder = thread::Builder::new()
        .name("toggle-recorder".into())
        .spawn(move || {
            let _stop = StopOnDrop(Arc::clone(&recorder_control));
            let result = record(
                receiver,
                audio_tx,
                options.device,
                options.max_recording,
                recorder_cues,
                Arc::clone(&recorder_control),
            );
            if result.is_err() {
                recorder_control.shutdown();
            }
            result
        })?;
    eprintln!(
        "Voco ready: {} toggles recording. Microphone is closed while idle.",
        options.shortcut
    );
    let ui_result = _hotkey.run_main_loop(&control);
    control.shutdown();
    // Wake the blocking Unix accept without a polling timer.
    let _ = control::request(&socket_path, "quit");
    let recorded = recorder.join().map_err(|_| "recorder worker panicked")?;
    control.inference.notify();
    let processed = processor.join().map_err(|_| "inference worker panicked")?;
    let served = socket_worker
        .join()
        .map_err(|_| "control worker panicked")?;
    sounds.finish();
    processed?;
    recorded?;
    served?;
    ui_result
}

struct Recording {
    stream: Option<cpal::Stream>,
    stop: Arc<AtomicBool>,
    samples: rtrb::Consumer<f32>,
    dropped: Arc<std::sync::atomic::AtomicUsize>,
    failed: Arc<AtomicBool>,
    resampler: StreamingResampler,
    audio: Vec<f32>,
    native: Vec<f32>,
    converted: Vec<f32>,
    maximum: usize,
    started: Instant,
}
impl Recording {
    fn start(device: Option<&str>, wake: Arc<Wake>, maximum: Duration) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let capture = Capture::open_quiet(device, Arc::clone(&stop), wake, true)?;
        capture.start()?;
        let (stream, samples, rate, dropped, failed) = capture.into_parts();
        Ok(Self {
            stream: Some(stream),
            stop,
            samples,
            dropped,
            failed,
            resampler: StreamingResampler::new(rate)?,
            audio: Vec::with_capacity(SAMPLE_RATE * 5),
            native: Vec::with_capacity(4096),
            converted: Vec::with_capacity(8192),
            maximum: maximum.as_secs() as usize * SAMPLE_RATE,
            started: Instant::now(),
        })
    }
    fn drain(&mut self) -> Result<()> {
        if self.failed.load(Ordering::Relaxed) {
            return Err("microphone stream failed".into());
        }
        if self.dropped.swap(0, Ordering::Relaxed) != 0 {
            return Err(
                "microphone overrun: recording discarded instead of typing incomplete speech"
                    .into(),
            );
        }
        loop {
            self.native.clear();
            while self.native.len() < 4096 {
                match self.samples.pop() {
                    Ok(sample) => self.native.push(sample),
                    Err(_) => break,
                }
            }
            if self.native.is_empty() {
                break;
            }
            self.converted.clear();
            self.resampler.push(&self.native, &mut self.converted)?;
            self.audio.extend_from_slice(
                &self.converted[..self
                    .converted
                    .len()
                    .min(self.maximum.saturating_sub(self.audio.len()))],
            );
        }
        Ok(())
    }
    fn finish(mut self) -> Result<Vec<f32>> {
        self.stop.store(true, Ordering::Release);
        drop(self.stream.take());
        self.drain()?;
        self.converted.clear();
        self.resampler.finish(&mut self.converted)?;
        self.audio.extend_from_slice(
            &self.converted[..self
                .converted
                .len()
                .min(self.maximum.saturating_sub(self.audio.len()))],
        );
        Ok(self.audio)
    }
}

fn record(
    receiver: Receiver<()>,
    sender: SyncSender<Vec<f32>>,
    device: Option<String>,
    maximum: Duration,
    cues: Cues,
    control: Arc<Control>,
) -> Result<()> {
    control.recorder.register();
    let mut recording: Option<Recording> = None;
    let mut finish_at = None;
    loop {
        while receiver.try_recv().is_ok() {
            if recording.is_some() {
                // One small tail catches the device's final callback and quiet endings.
                if finish_at.is_none() {
                    finish_at = Some(Instant::now() + Duration::from_millis(80));
                }
            } else if control.state.load(Ordering::Acquire) == IDLE
                && !control.stop.load(Ordering::Acquire)
            {
                recording = Some(Recording::start(
                    device.as_deref(),
                    Arc::clone(&control.recorder),
                    maximum,
                )?);
                control.state.store(RECORDING, Ordering::Release);
                cues.play(Cue::Start);
                control.warmup.store(true, Ordering::Release);
                control.inference.notify();
            }
        }
        if let Some(active) = &mut recording {
            active.drain()?;
            let finished = control.stop.load(Ordering::Acquire)
                || finish_at.is_some_and(|end| Instant::now() >= end)
                || active.started.elapsed() >= maximum;
            if finished {
                let audio = recording.take().expect("recording active").finish()?;
                finish_at = None;
                // Explicit recordings are not VAD-sliced or rejected for short words.
                if !audio.is_empty() && audio.iter().any(|&x| x != 0.0) {
                    control.state.store(TRANSCRIBING, Ordering::Release);
                    sender.send(audio)?;
                } else {
                    control.state.store(IDLE, Ordering::Release);
                    cues.play(Cue::Done);
                }
                control.inference.notify();
            }
        }
        if control.stop.load(Ordering::Acquire) {
            break;
        }
        let timeout = recording.as_ref().map(|active| {
            let cap = maximum.saturating_sub(active.started.elapsed());
            finish_at.map_or(cap, |end| {
                cap.min(end.saturating_duration_since(Instant::now()))
            })
        });
        control.recorder.wait(timeout);
    }
    drop(sender);
    control.inference.notify();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn inference(
    receiver: Receiver<Vec<f32>>,
    model_dir: PathBuf,
    timeout: Option<Duration>,
    eager: bool,
    print_only: bool,
    cues: Cues,
    control: Arc<Control>,
    ready: SyncSender<bool>,
) -> Result<()> {
    control.inference.register();
    let mut output = match DictationOutput::new(print_only, false) {
        Ok(output) => output,
        Err(error) => {
            let _ = ready.send(false);
            return Err(error);
        }
    };
    let mut model = ModelCache::new(timeout);
    if eager && let Err(error) = model.get_or_load(|| Parakeet::load(&model_dir)) {
        let _ = ready.send(false);
        return Err(error);
    }
    let mut preprocessor = Preprocessor::new();
    let _ = ready.send(true);
    loop {
        match receiver.try_recv() {
            Ok(audio) => {
                let features = preprocessor.compute(&audio)?;
                let text = model
                    .get_or_load(|| Parakeet::load(&model_dir))?
                    .transcribe(features)?;
                // Do not turn text into Ctrl/Alt shortcuts while the activation chord is held.
                while control.keys_down.load(Ordering::Acquire)
                    && !control.stop.load(Ordering::Acquire)
                {
                    control.inference.wait(None);
                }
                if !text.is_empty() {
                    output.emit(text)?;
                }
                model.touch(Instant::now());
                control.state.store(IDLE, Ordering::Release);
                cues.play(Cue::Done);
                control.recorder.notify();
            }
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {
                if control.warmup.swap(false, Ordering::AcqRel)
                    && !control.stop.load(Ordering::Acquire)
                {
                    model.get_or_load(|| Parakeet::load(&model_dir))?;
                }
                let active = control.state.load(Ordering::Acquire) != IDLE;
                let now = Instant::now();
                model.expire(now, active);
                control.inference.wait(model.wait_timeout(now, active));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "manual_tests.rs"]
mod tests;
