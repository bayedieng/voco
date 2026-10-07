# Voco

Local, offline voice dictation with Parakeet TDT ONNX models. **Press once to
record; press again to stop and type the complete recording.** Nothing is
streamed to a server, and the microphone is closed while idle.

## Quick start

Use release builds for speech recognition:

```sh
cargo run --release -- --daemon --print-only  # safe test: prints, never types
cargo run --release -- --daemon              # type into the focused application
cargo run --release -- --list-devices
cargo run --release -- --preview-cues        # hear the two notification sounds
cargo run --release -- audio.wav             # file transcription; never types
```

The default shortcut is **Ctrl+Alt+Z**, reachable with the left hand. On macOS,
it is **Control+Option+Z**. It is configurable with `--hotkey "Ctrl+Alt+X"`.

1. Focus the intended text field.
2. Press and release the shortcut. Wait for the short start click, then speak.
3. Press and release it again. The microphone closes, the whole clip is
   transcribed, and Enigo inserts plain text plus a trailing space. **Enter is
   never pressed.** A different short click indicates completion.

There is no VAD, silence-based endpointing, minimum speech duration, or pause
splitting: manual recording boundaries replace them. Short/quiet speech is not
filtered out. Stop retains an **80 ms tail** for the microphone's final callback
and quiet word endings, then flushes the resampler. Exactly silent recordings
produce no text; otherwise ASR receives the full captured clip, including noise
and pauses. The model can still misrecognize speech or hallucinate on noise.

Wait for completion before recording again. Toggles during transcription are
ignored; recordings cannot accumulate in an unbounded queue. By default a
recording automatically stops at **60 seconds**; `--max-recording-secs` accepts
1–120 seconds and bounds memory use. This is offline, whole-recording ASR, not
live transcription.

`--mic` is an alias for `--daemon`; the old continuous/VAD mode and VAD flags
have been removed. WAV mode still accepts `audio.wav model-directory`.
`--device "exact device name"` selects an input device; `--model-dir PATH`
selects the four Parakeet files.

## Install a permanent user-session daemon

Run from this repository **as your desktop user, not with sudo**. Deployment and
the daemon refuse root execution. No step invokes `sudo` or silently escalates
privileges; if system dependencies need administrator installation, that is a
separate, explicitly approved user action before deployment.

```sh
cargo xtask service install --dry-run    # inspect paths/unit; no changes or startup
cargo xtask service install              # release build, deploy, enable login startup, start
cargo xtask service status
```

Deployment uses the `service-manager` crate, **systemd user services on Linux**
and a **LaunchAgent on macOS**, not a root/system daemon. A system service would
not have the right microphone, graphical-session or input-injection permissions.
The executable and models are copied outside the build tree, so subsequent
`cargo clean` operations do not break the installed service.

### Permanently load / unload

**Run `cargo xtask service install` once first.** `load` only re-enables an existing
installation; it does not deploy the executable/models or create a service file.
`install --dry-run` is only a preview. If you see “Unit voco-vocod.service does not
exist”, run `install` (without `--dry-run`), then check `service status`.

```sh
cargo xtask service unload      # stop now AND disable future login startup
cargo xtask service load        # enable future login startup AND start now
cargo xtask service uninstall   # remove service registration entirely
```

**Unload stays unloaded across logins/reboots until load is run.** Unload retains
the service configuration, binary and models so load does not rebuild/download
anything. Uninstall also retains deployment files; delete the deployment directory
manually if you no longer want them.

For temporary control, without changing login startup:

```sh
cargo xtask service stop
cargo xtask service start
cargo xtask service restart
cargo xtask service toggle      # same recording toggle as the shortcut
```

After a permanent unload, use **load**, especially on macOS where launchd's
persistent disabled policy prevents starting a disabled agent. `vocod --quit`
is a clean process exit, **not** permanent unload. Reinstall updates the program,
models and service options and re-enables it unless `--no-start` is specified.

Install accepts `--device`, `--model-dir` (model source to copy), `--hotkey`,
`--external-hotkey`, `--max-recording-secs`, `--model-idle-secs`,
`--keep-model-loaded`, `--no-cues`,
`--print-only`, `--skip-build`, and `--no-start`. For example:

```sh
cargo xtask service install --keep-model-loaded --hotkey "Ctrl+Alt+X"
```

The daemon runs in the foreground internally; the native service manager handles
background execution, login startup and restart-on-failure. Ctrl+C, SIGTERM and
SIGHUP stop it cleanly, closing capture and finishing an active recording before
exit. **Stopping can still insert the final recording into the focused field.**

### Linux

- Build dependencies include ALSA headers (`libasound2-dev` on Debian/Ubuntu,
  `alsa-lib-devel` on Fedora) and the desktop's runtime libraries.
- Deployment defaults to `~/.local/share/voco` (or `$XDG_DATA_HOME/voco`).
  Unit: `~/.config/systemd/user/voco-vocod.service`. A hidden desktop entry under
  the user data directory identifies Voco to the shortcut permission portal.
- `xtask` imports the current desktop environment into the user service manager.
  Install/load from a terminal in the intended desktop session. Autostart is tied
  to `graphical-session.target`; custom compositors must activate that target
  (e.g. via UWSM), or start the service from their login configuration after
  importing the session environment. Do not enable user lingering for this app.
- X11 uses native global hotkeys. Wayland uses the consent-based XDG
  **GlobalShortcuts portal**; your desktop may display a shortcut approval dialog
  or override the preferred combination. Portal availability varies.
- If the portal is unavailable, configure a desktop/compositor shortcut executing
  `~/.local/share/voco/bin/vocod --toggle`, then install with `--external-hotkey`.
  Prefer a key-release binding if supported; release modifier keys before text
  insertion. For Hyprland, for example:
  `bindr = CTRL ALT, Z, exec, ~/.local/share/voco/bin/vocod --toggle`.
- Shortcut registration and typing permissions are separate. Enigo needs input
  injection support from your compositor; a working portal alone does not grant
  it. If unsupported, use `--print-only` or a supported desktop/session.
- On Wayland desktops without Enigo's native keyboard protocols (such as KDE),
  Voco keeps Enigo's XWayland fallback but types existing XKB keycodes with Shift.
  This avoids dynamic key remapping that can drop capitals and shifted punctuation
  (e.g. `This` becoming `his`). The active layout and Caps Lock are read once per
  dictation; no idle polling, clipboard changes or elevated permissions are added.
  Key presses/releases are paced at 5 ms each, with 20 ms for Shift changes, to
  avoid delayed modifier updates causing random capitalization. This adds about
  10 ms per character plus 40 ms per shifted character to typing time.
  Symbols absent from the layout still use Enigo's compositor-dependent Unicode
  fallback. Native Wayland and macOS typing are unchanged. To separate recognition
  errors from injection errors, test a foreground daemon with `--print-only`.
- Logs: `journalctl --user -u voco-vocod.service`.

### macOS

- Deployment: `~/Library/Application Support/voco/Voco.app` and sibling `models`.
  LaunchAgent: `~/Library/LaunchAgents/com.voco.vocod.plist`.
- The app bundle includes a microphone usage description, stable bundle identifier
  and ad-hoc signature. Native hotkeys and the event loop run on the main thread.
- Authorize **Microphone** and **Accessibility** in System Settings → Privacy &
  Security. For an initial interactive permission check:

  ```sh
  cargo xtask service install --no-start
  cargo xtask service run       # deployed app, in the foreground; trigger recording
  # Grant permissions, exit with Ctrl+C, then:
  cargo xtask service load
  ```

  Grant the responsible app macOS actually lists (Voco, or Terminal during a
  terminal-launched check); a LaunchAgent may need its own authorization. You can
  add Voco.app explicitly to Accessibility. Ad-hoc signatures can require renewed
  authorization after updates; a stable Developer ID signature is preferable for
  production. Check background-item permission if macOS blocks login startup.
- Logs: `~/Library/Application Support/voco/logs/{stdout,stderr}.log`. These files
  are not automatically rotated; diagnostic `--print-only` output can grow them.
- macOS service definitions are unit-tested, but live shortcut, permission and
  capture behavior has **not been verified on a Mac** in this development environment.

## Background resources and latency

The default policy favors low idle usage:

- Idle workers block on events; no continuous microphone, VAD, resampling or
  short-interval polling runs. Sound devices exist only during cue playback.
- ASR loads lazily when recording starts, overlapping model loading with speech.
  ONNX sessions are reused for later recordings and freed after **60 idle seconds**
  (`--model-idle-secs`). Eviction never occurs during recording/transcription.
- Eviction also asks the system allocator to return freed pages on glibc Linux
  and macOS, rather than retaining large free heaps in RSS.
- ONNX workers stop spinning between requests, while keeping inference parallelism.
  Capture uses a lock-free, two-second ring and a separate worker; the audio
  callback performs no allocation, inference, logging or keyboard work.

**Keeping the daemon enabled is different from keeping the model in RAM.**
`--keep-model-loaded` preloads ASR and disables eviction, minimizing first-word
latency at a substantial memory cost. Default mode preserves warm inference
speed, but a short recording after eviction can finish before cold loading does;
that remaining load time increases stop-to-text latency. There is no way to
unload the weights and simultaneously guarantee resident-model cold latency.

Example Linux measurements on this development host (not guarantees):

| Check | Result |
| --- | --- |
| Idle daemon, no model loaded (`--external-hotkey --no-cues --print-only`) | ~11 MB RSS; 0 CPU ticks over 5 s |
| Cold ASR load | ~2.4 s |
| Warm inference, bundled ~8 s speech clip | ~0.8–1.0 s |
| Loaded-model idle | 0 CPU ticks over 2 s |
| ASR test RSS after warm runs / after eviction + heap reclamation | ~1.1 GB / ~37 MB |

The ASR measurements are an isolated test, not end-to-end typing timings. Actual
latency includes the 80 ms tail, preprocessing, remaining cold load and desktop
insertion. Hardware, desktop backends and model choice affect the results.

## Sounds, privacy and implementation

The start (`click4.wav`) and completion (`click3.wav`) cues are from
[Kenney UI Audio](https://kenney.nl/assets/ui-audio), **CC0**, played at reduced
gain. `build.rs` fetches a pinned mirror revision, verifies SHA-256 hashes and
embeds the short WAVs into the executable. They require no runtime download.
At build time it also downloads missing Parakeet models; runtime recognition is
local. See `licenses/kenney-ui-audio-LICENSE.md`. Use `--no-cues` to disable audio.

Recordings stay in memory; no microphone audio files are saved. Normal daemon
mode does not print/store transcripts. `--print-only` intentionally prints them
(and a service manager can retain that output in logs). Control commands use a
private per-user Unix socket (directory mode 0700, socket 0600), with one daemon
per socket:

```sh
vocod --toggle
vocod --daemon-status          # idle / recording / transcribing
vocod --quit
```

Use the deployed binary's full path if it is not on `PATH`. `--control-socket`
overrides the path for both server and client, and requires a private directory.

- `src/manual.rs`: recording state machine, model prefetch, inference/output workers.
- `src/control.rs`, `src/hotkey.rs`: private IPC and platform shortcut registration.
- `src/microphone.rs`, `src/audio.rs`: capture, WAV loading, band-limited resampling.
- `src/preprocessing.rs`, `src/model.rs`: Mel frontend and greedy Parakeet TDT ASR.
- `src/model_cache.rs`, `src/wake.rs`: idle memory reclamation and event-driven wakeups.
- `src/output.rs`, `src/sound.rs`: safe text insertion and embedded cues.
- `xtask/src/`: deployment and user-session service lifecycle.

Microphone failure/overrun discards the interrupted recording and reports an
error instead of silently typing incomplete speech. Capture/inference failure
exits for the service manager to handle. Models are not promised perfect accuracy;
test in print-only mode before enabling injection into sensitive applications.

## Tests

```sh
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --release model::tests -- --ignored --nocapture --test-threads=1
cargo xtask service install --dry-run
```

Tests cover preprocessing reference values, chunk-independent resampling, full
recording/final-sample preservation, short quiet input, capture gaps, key-repeat
suppression, model reuse/eviction, wakeups, embedded WAVs, deployment and service
rendering. Real microphone capture, shortcuts and typing need a desktop smoke test.
