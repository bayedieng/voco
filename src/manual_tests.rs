use super::*;
fn recording(samples: &[f32], rate: usize) -> Recording {
    let (mut producer, consumer) = rtrb::RingBuffer::new(samples.len().max(1));
    for &sample in samples {
        producer.push(sample).unwrap();
    }
    Recording {
        stream: None,
        stop: Arc::default(),
        samples: consumer,
        dropped: Arc::default(),
        failed: Arc::default(),
        resampler: StreamingResampler::new(rate).unwrap(),
        audio: Vec::new(),
        native: Vec::with_capacity(4096),
        converted: Vec::new(),
        maximum: SAMPLE_RATE * 60,
        started: Instant::now(),
    }
}
#[test]
fn quiet_short_recording_and_final_callback_are_not_filtered() -> Result<()> {
    let mut samples = vec![0.0001; 5000];
    samples[0] = 0.5;
    samples[4999] = -0.5;
    assert_eq!(recording(&samples, 16000).finish()?, samples);
    assert_eq!(
        recording(&[0.0001; 320], 16000).finish()?,
        vec![0.0001; 320]
    );
    Ok(())
}
#[test]
fn resampling_is_finished_and_gaps_are_rejected() -> Result<()> {
    let samples = vec![0.1; 4800];
    let audio = recording(&samples, 48000).finish()?;
    assert!((audio.len() as isize - 1600).abs() <= 1);
    let interrupted = recording(&samples, 48000);
    interrupted.dropped.store(32, Ordering::Relaxed);
    assert!(interrupted.finish().is_err());
    Ok(())
}
