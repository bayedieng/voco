//! Bounded capture -> endpointing -> inference -> keyboard pipeline.
//! CPAL never waits for inference or Enigo. ONNX and FFT plans remain warm.
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, SyncSender, TrySendError},
    },
    thread,
    time::{Duration, Instant},
};

use rtrb::Consumer;

use crate::{
    Result,
    audio::StreamingResampler,
    microphone::Capture,
    model::Parakeet,
    output::DictationOutput,
    preprocessing::Preprocessor,
    utterance::{EndpointConfig, Segmenter},
    vad::{SileroVad, SpeechDetector},
};

pub fn run(
    model_dir: &Path,
    vad_path: &Path,
    device: Option<&str>,
    config: EndpointConfig,
    print_only: bool,
) -> Result<()> {
    let config = config.validate()?;
    let mut model = Parakeet::load(model_dir)?;
    let mut preprocessor = Preprocessor::new();
    let mut output = DictationOutput::new(print_only)?;
    let segmenter = Segmenter::new(config, SileroVad::load(vad_path)?)?;
    let stop = Arc::new(AtomicBool::new(false));
    let signal_stop = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        signal_stop.store(true, Ordering::Relaxed);
    })?;
    let capture = Capture::open(device, Arc::clone(&stop))?;
    capture.start()?;
    let (stream, samples, rate, dropped, failed) = capture.into_parts();
    // At most four completed utterances can wait behind inference; don't grow indefinitely.
    let (sender, receiver) = mpsc::sync_channel(4);
    let worker_stop = Arc::clone(&stop);
    let worker = thread::Builder::new()
        .name("audio-endpointing".into())
        .spawn(move || {
            capture_worker(
                samples,
                rate,
                dropped,
                failed,
                worker_stop,
                segmenter,
                sender,
            )
        })?;
    eprintln!(
        "Listening. {} Ctrl+C stops capture and finishes pending utterances.",
        if print_only {
            "Printing only."
        } else {
            "Text is typed into the currently focused application."
        }
    );
    let result = (|| -> Result<()> {
        // The worker closes the channel on shutdown, after flushing the active utterance.
        while let Ok(audio) = receiver.recv() {
            let start = Instant::now();
            let features = preprocessor.compute(&audio)?;
            let text = model.transcribe(features)?;
            if !text.is_empty() {
                output.emit(text)?;
                eprintln!(
                    "Transcribed in {:.0}ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        Ok(())
    })();
    // Always stop and join the worker, including on model or keyboard failures.
    stop.store(true, Ordering::Relaxed);
    drop(stream);
    let worker_result = worker.join().map_err(|_| "audio worker panicked")?;
    result?;
    worker_result
}

fn capture_worker<D: SpeechDetector>(
    mut samples: Consumer<f32>,
    rate: usize,
    dropped: Arc<AtomicUsize>,
    failed: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    mut segmenter: Segmenter<D>,
    sender: SyncSender<Vec<f32>>,
) -> Result<()> {
    let mut resampler = StreamingResampler::new(rate)?;
    let mut native = Vec::with_capacity(4096);
    let mut converted = Vec::with_capacity(8192);
    loop {
        if failed.load(Ordering::Relaxed) {
            return Err("microphone stream failed; check device connection".into());
        }
        let lost = dropped.swap(0, Ordering::Relaxed);
        if lost > 0 {
            eprintln!("Warning: lost {lost} microphone samples; discarding interrupted utterance.");
            // Drop queued audio and filter history so no utterance crosses a gap.
            let queued = samples.slots();
            for _ in 0..queued {
                let _ = samples.pop();
            }
            segmenter.reset();
            resampler = StreamingResampler::new(rate)?;
        }
        native.clear();
        while native.len() < 4096 {
            match samples.pop() {
                Ok(sample) => native.push(sample),
                Err(_) => break,
            }
        }
        if !native.is_empty() {
            converted.clear();
            resampler.push(&native, &mut converted)?;
            segmenter.push(&converted, |audio| enqueue(&sender, audio))?;
        } else if stop.load(Ordering::Relaxed) {
            break;
        } else {
            thread::sleep(Duration::from_millis(5));
        }
    }
    converted.clear();
    resampler.finish(&mut converted)?;
    segmenter.push(&converted, |audio| enqueue(&sender, audio))?;
    if let Some(audio) = segmenter.finish()? {
        enqueue(&sender, audio);
    }
    Ok(())
}

fn enqueue(sender: &SyncSender<Vec<f32>>, audio: Vec<f32>) {
    match sender.try_send(audio) {
        Ok(()) | Err(TrySendError::Disconnected(_)) => {}
        Err(TrySendError::Full(_)) => eprintln!(
            "Warning: inference backlog is full; dropping an utterance instead of blocking capture."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::FRAME_SAMPLES;
    use rtrb::RingBuffer;

    struct FixedDetector;
    impl SpeechDetector for FixedDetector {
        fn probability(&mut self, _: &[f32; FRAME_SAMPLES]) -> Result<f32> {
            Ok(0.9)
        }
        fn reset(&mut self) {}
    }
    fn test_segmenter() -> Segmenter<FixedDetector> {
        Segmenter::new(EndpointConfig::default(), FixedDetector).unwrap()
    }

    #[test]
    fn worker_drains_resamples_and_flushes_on_shutdown() -> Result<()> {
        for rate in [16000, 48000] {
            let (mut producer, samples) = RingBuffer::new(rate);
            for _ in 0..rate / 2 {
                producer.push(0.1).unwrap();
            }
            let (sender, receiver) = mpsc::sync_channel(4);
            capture_worker(
                samples,
                rate,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicBool::new(false)),
                Arc::new(AtomicBool::new(true)),
                test_segmenter(),
                sender,
            )?;
            let utterances: Vec<_> = receiver.into_iter().collect();
            assert_eq!(utterances.len(), 1);
            assert!(utterances[0].len() >= 7000);
            assert!(utterances[0].iter().all(|x| x.is_finite()));
        }
        Ok(())
    }

    #[test]
    fn neural_vad_gates_silence_and_tones_before_inference() -> Result<()> {
        let vad = SileroVad::load(Path::new(crate::vad::DEFAULT_VAD_PATH))?;
        let mut segmenter = Segmenter::new(EndpointConfig::default(), vad)?;
        let audio: Vec<_> = (0..32000)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 16000.0).sin() * 0.2)
            .collect();
        for chunk in audio.chunks(137) {
            segmenter.push(chunk, |_| panic!("non-speech reached inference queue"))?;
        }
        segmenter.push(&vec![0.0; 32000], |_| {
            panic!("silence reached inference queue")
        })?;
        assert!(segmenter.finish()?.is_none());
        Ok(())
    }

    #[test]
    #[ignore = "VAD -> Parakeet speech integration: cargo test --release -- --ignored"]
    fn neural_vad_utterances_transcribe_bundled_speech() -> Result<()> {
        let audio = crate::audio::read_wav(
            &Path::new(crate::model::DEFAULT_MODEL_DIR).join("test_wavs/0.wav"),
        )?;
        let vad = SileroVad::load(Path::new(crate::vad::DEFAULT_VAD_PATH))?;
        let mut segmenter = Segmenter::new(EndpointConfig::default(), vad)?;
        let mut utterances = Vec::new();
        for chunk in audio.chunks(137) {
            segmenter.push(chunk, |audio| utterances.push(audio))?;
        }
        segmenter.push(&vec![0.0; 16000], |audio| utterances.push(audio))?;
        if let Some(audio) = segmenter.finish()? {
            utterances.push(audio);
        }
        assert!(!utterances.is_empty(), "speech was not detected");
        let mut model = Parakeet::load(Path::new(crate::model::DEFAULT_MODEL_DIR))?;
        let mut preprocessor = Preprocessor::new();
        let mut text = String::new();
        for audio in utterances {
            let start = Instant::now();
            let features = preprocessor.compute(&audio)?;
            let preprocessing = start.elapsed();
            let start = Instant::now();
            text.push_str(&model.transcribe(features)?);
            text.push(' ');
            eprintln!(
                "Speech context: {:.2}s | preprocessing: {:.1}ms | inference: {:.1}ms",
                audio.len() as f64 / crate::audio::SAMPLE_RATE as f64,
                preprocessing.as_secs_f64() * 1000.0,
                start.elapsed().as_secs_f64() * 1000.0,
            );
        }
        eprintln!("VAD-gated transcript: {text}");
        assert!(text.contains("I don't wish to see it any more"), "{text}");
        assert!(text.contains("the old portrait"), "{text}");
        Ok(())
    }

    #[test]
    fn stream_errors_propagate() {
        let (_, samples) = RingBuffer::new(16);
        let (sender, _) = mpsc::sync_channel(4);
        assert!(
            capture_worker(
                samples,
                16000,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicBool::new(true)),
                Arc::new(AtomicBool::new(false)),
                test_segmenter(),
                sender
            )
            .is_err()
        );
    }

    #[test]
    fn full_inference_queue_does_not_block_capture() {
        let (sender, receiver) = mpsc::sync_channel(1);
        enqueue(&sender, vec![0.1]);
        enqueue(&sender, vec![0.2]);
        assert_eq!(receiver.try_recv().unwrap(), [0.1]);
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn overrun_discards_interrupted_audio() -> Result<()> {
        let (mut producer, samples) = RingBuffer::new(16000);
        for _ in 0..16000 {
            producer.push(0.1).unwrap();
        }
        let (sender, receiver) = mpsc::sync_channel(4);
        capture_worker(
            samples,
            16000,
            Arc::new(AtomicUsize::new(100)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(true)),
            test_segmenter(),
            sender,
        )?;
        assert!(receiver.into_iter().next().is_none());
        Ok(())
    }
}
