# Voco

Local Parakeet ONNX transcription and utterance-based microphone dictation.

## Usage

Use a release build for real-time performance:

```sh
# Transcribe a file; this only prints text.
cargo run --release -- audio.wav

# List microphone names.
cargo run --release -- --list-devices

# Test microphone recognition without typing.
cargo run --release -- --mic --print-only

# Dictate into the currently focused application using Enigo.
cargo run --release -- --mic

# Optional device/model selection and endpoint tuning.
cargo run --release -- --mic --device "exact device name" \
  --model-dir models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 \
  --vad-threshold 0.5 --silence-ms 600 --min-speech-ms 96 --max-utterance-secs 15
```

Focus the intended text field after starting dictation. Each completed utterance is
inserted as plain text with a trailing space; **Enter is never pressed**. Ctrl+C
stops capture and processes queued/unfinished valid utterances before exiting.
WAV mode also accepts the legacy `audio.wav model-directory` arguments.

Microphone capture is continuous, but these models are **offline**, not streaming
ASR: decoding happens after an utterance ends. **Silero VAD** classifies speech
before Parakeet runs; non-speech does not get transcribed or typed. Default
endpointing uses 32ms neural-VAD frames, 500ms pre-roll, a 64ms confirmed onset,
at least 96ms of speech, and 600ms of non-speech before emitting an utterance.
The full captured non-speech hangover is retained for ASR: VAD probabilities are
not precise phoneme boundaries, and trimming it could remove quiet word endings.
Longer pre-roll protects initial consonants when VAD confidence arrives late;
the shorter speech minimum avoids dropping brief words. These changes do not
increase the endpoint wait or the number of VAD evaluations.
A 0.5 start probability and 0.35 release probability provide hysteresis, preserving
uncertain/quiet speech within an active utterance. Long utterances are split at
15 seconds to bound memory and latency. VAD is not perfect: background speech,
singing, and some noises may still trigger it.

Raise `--vad-threshold` (e.g. 0.65) to reject more noise or lower it (e.g. 0.4) if
speech is missed. `--threshold` remains an alias, but **now means speech probability,
not RMS amplitude**; old values such as 0.01 should not be reused. Use `--print-only`
to tune it without injecting keystrokes.

The [Silero VAD](https://github.com/snakers4/silero-vad) model is MIT-licensed
(license: `licenses/silero-vad-LICENSE.md`). We pin **v6.2.3**, a 2.3MB ONNX file,
downloaded and SHA-256-verified by `build.rs` into `models/silero-vad-v6.2.3.onnx`.
Its upstream documentation reports sub-millisecond inference per 30+ms chunk on
a single CPU thread; measure your machine using the benchmark below. The default
model requires no new runtime dependency beyond our existing ONNX Runtime.
`--vad-model path/to/model.onnx` overrides the path and must use the same Silero
streaming interface; arbitrary ONNX VAD models are not interchangeable.

CPAL uses the default host/device configuration and first input channel. Audio is
converted to normalized float samples and continuously resampled to 16kHz using
a band-limited FFT resampler. No audio files are written by microphone mode.

On Linux, building CPAL requires ALSA development headers (`libasound2-dev` on
Debian/Ubuntu, `alsa-lib-devel` on Fedora). Enigo needs a usable desktop session
and input-injection permissions. Wayland support depends on compositor protocols;
`--print-only` does not require an Enigo connection. macOS may require microphone
and Accessibility permissions; Windows may require microphone permission.

## Organization and performance

- `src/main.rs`, `src/cli.rs`: entry point, WAV mode, and configuration.
- `src/audio.rs`: WAV loading and reusable continuous resampling.
- `src/microphone.rs`: CPAL capture into a bounded, lock-free SPSC ring.
- `src/vad.rs`: one-thread Silero ONNX inference, rolling context and recurrent state.
- `src/utterance.rs`: probability hysteresis and speech-only utterance segmentation.
- `src/preprocessing.rs`: reusable FFT buffers and sparse precomputed Mel filters.
- `src/model.rs`: persistent ONNX sessions and greedy TDT decoding.
- `src/output.rs`: printing and Enigo text insertion.
- `src/dictation.rs`: capture/endpoint worker and inference/output orchestration.

The real-time callback does no allocation, logging, resampling, inference, or
keyboard work. Endpointing runs on a separate worker, so inference cannot stall
capture. The ring holds two seconds of native-rate mono samples and the inference
queue holds four utterances. Overruns discard the interrupted utterance; a full
inference queue drops new utterances with a warning instead of growing memory or
blocking capture. Models and preprocessing plans are loaded once per mic session.
Silero runs on the audio worker using reusable waveform/context/state buffers,
not in the real-time callback. It consumes 512 new samples plus 64 context samples
at 16kHz, with recurrent state retained between frames/utterances and reset on capture
gaps. While idle, only capture, resampling, and the small VAD run; Parakeet and Enigo
receive nothing. A final partial VAD frame is padded for classification only.

## Tests

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --ignored # ASR and VAD -> ASR integration tests
cargo test --release vad::tests::benchmark -- --ignored --nocapture # local VAD timing
```

Unit tests cover preprocessing reference values, continuous resampling,
endpointing, capture gaps, shutdown flushing, CLI modes, and queue backpressure.
Silero tests compare probabilities against the upstream Python wrapper and check
that silence/tones do not reach the transcription queue. Regression tests also
cover delayed VAD onset, quiet trailing consonants, and short confirmed words.
Hardware microphone capture and desktop text injection need a live-device smoke test.

