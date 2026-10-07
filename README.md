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
  --threshold 0.01 --silence-ms 600 --min-speech-ms 200 --max-utterance-secs 15
```

Focus the intended text field after starting dictation. Each completed utterance is
inserted as plain text with a trailing space; **Enter is never pressed**. Ctrl+C
stops capture and processes queued/unfinished valid utterances before exiting.
WAV mode also accepts the legacy `audio.wav model-directory` arguments.

Microphone capture is continuous, but these models are **offline**, not streaming
ASR: decoding happens after an utterance ends. Default endpointing uses 20ms RMS
frames, 200ms pre-roll, a 60ms onset, at least 200ms of activity, and 600ms of quiet.
Long utterances are split at 15 seconds to bound memory and latency. This is a
lightweight energy detector, not a neural VAD: fans, music, and keyboard noise can
trigger it. Raise `--threshold` for noisy rooms or lower it for a quiet microphone.
Use `--print-only` to tune it without injecting keystrokes.

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
- `src/utterance.rs`: activity detection and utterance segmentation.
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

## Tests

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release -- --ignored # bundled WAV / ONNX integration test
```

Unit tests cover preprocessing reference values, continuous resampling,
endpointing, capture gaps, shutdown flushing, CLI modes, and queue backpressure.
Hardware microphone capture and desktop text injection need a live-device smoke test.

