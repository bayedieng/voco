//! Per-user deployment. Models stay on disk, not in a root-owned service or build tree.
use crate::{ROOT, Result, service::Platform};
use directories::BaseDirs;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

pub struct Layout {
    pub root: PathBuf,
    pub program: PathBuf,
    pub model_dir: PathBuf,
    pub logs: PathBuf,
    pub bundle: PathBuf,
}

impl Layout {
    pub fn for_user(platform: Platform) -> Result<Self> {
        let base = BaseDirs::new().ok_or("cannot determine user directories")?;
        Ok(Self::at(base.data_local_dir().join("voco"), platform))
    }

    pub fn at(root: PathBuf, platform: Platform) -> Self {
        let bundle = root.join("Voco.app");
        let program = match platform {
            Platform::Macos => bundle.join("Contents/MacOS/vocod"),
            Platform::Linux => root.join("bin/vocod"),
        };
        Self {
            program,
            model_dir: root.join("models/parakeet"),
            logs: root.join("logs"),
            bundle,
            root,
        }
    }
}

pub fn build(root: &Path) -> Result<PathBuf> {
    let mut child = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .current_dir(root)
        .args([
            "build",
            "--release",
            "-p",
            "vocod",
            "--message-format=json-render-diagnostics",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let mut executable = None;
    for line in BufReader::new(child.stdout.take().ok_or("missing cargo output")?).lines() {
        let line = line?;
        if let Ok(message) = serde_json::from_str::<serde_json::Value>(&line) {
            if message["reason"] == "compiler-artifact"
                && message["target"]["name"] == "vocod"
                && let Some(path) = message["executable"].as_str()
            {
                executable = Some(PathBuf::from(path));
            }
            if let Some(rendered) = message["message"]["rendered"].as_str() {
                eprint!("{rendered}");
            }
        }
    }
    if !child.wait()?.success() {
        return Err("release build failed".into());
    }
    executable.ok_or_else(|| "cargo did not report a vocod executable".into())
}

pub fn existing_release(root: &Path) -> Result<PathBuf> {
    let output = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .current_dir(root)
        .args(["metadata", "--no-deps", "--format-version=1"])
        .output()?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned().into());
    }
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let dir = metadata["target_directory"]
        .as_str()
        .ok_or("missing cargo target_directory")?;
    Ok(Path::new(dir).join("release/vocod"))
}

pub fn copy_release(
    binary: &Path,
    models: &Path,
    layout: &Layout,
    platform: Platform,
) -> Result<()> {
    for file in [
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ] {
        if !models.join(file).is_file() {
            return Err(format!("missing model file: {}", models.join(file).display()).into());
        }
    }
    if !binary.is_file() {
        return Err("release executable is missing".into());
    }
    fs::create_dir_all(&layout.root)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&layout.root, fs::Permissions::from_mode(0o700))?;
    }
    fs::create_dir_all(&layout.logs)?;
    copy_atomic(binary, &layout.program)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&layout.program, fs::Permissions::from_mode(0o755))?;
    }
    for file in [
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ] {
        copy_atomic(&models.join(file), &layout.model_dir.join(file))?;
    }
    copy_atomic(
        &Path::new(ROOT).join("licenses/kenney-ui-audio-LICENSE.md"),
        &layout.root.join("licenses/kenney-ui-audio-LICENSE.md"),
    )?;
    if platform == Platform::Linux {
        // The portal needs host-app metadata to identify Voco in its consent dialog.
        let applications = layout
            .root
            .parent()
            .ok_or("missing user data directory")?
            .join("applications");
        fs::create_dir_all(&applications)?;
        fs::write(
            applications.join(format!("{}.desktop", crate::LABEL)),
            format!(
                "[Desktop Entry]\nType=Application\nName=Voco\nComment=Local toggle voice dictation\nExec=systemctl --user start {}\nIcon=audio-input-microphone\nTerminal=false\nNoDisplay=true\n",
                crate::service::linux_unit_name()?
            ),
        )?;
    }
    if platform == Platform::Macos {
        let info = bundle_info();
        info.to_file_xml(layout.bundle.join("Contents/Info.plist"))?;
        let status = Command::new("codesign")
            .args([
                "--force",
                "--deep",
                "--sign",
                "-",
                "--identifier",
                crate::LABEL,
            ])
            .arg(&layout.bundle)
            .status()?;
        if !status.success() {
            return Err("macOS app bundle signing failed".into());
        }
    }
    Ok(())
}

fn copy_atomic(source: &Path, destination: &Path) -> Result<()> {
    let metadata = fs::metadata(source)?;
    if let Ok(existing) = fs::metadata(destination)
        && existing.len() == metadata.len()
        && existing.modified()? == metadata.modified()?
    {
        return Ok(());
    }
    fs::create_dir_all(
        destination
            .parent()
            .ok_or("missing destination directory")?,
    )?;
    let temporary = destination.with_extension(format!("{}.part", std::process::id()));
    fs::copy(source, &temporary)?;
    fs::File::open(&temporary)?.set_modified(metadata.modified()?)?;
    fs::rename(&temporary, destination)?;
    Ok(())
}

fn bundle_info() -> plist::Value {
    let mut info = plist::Dictionary::new();
    for (key, value) in [
        ("CFBundleIdentifier", crate::LABEL),
        ("CFBundleName", "Voco"),
        ("CFBundleExecutable", "vocod"),
        ("CFBundlePackageType", "APPL"),
        ("CFBundleVersion", env!("CARGO_PKG_VERSION")),
        (
            "NSMicrophoneUsageDescription",
            "Voco listens for speech to dictate text into your selected application.",
        ),
    ] {
        info.insert(key.into(), plist::Value::String(value.into()));
    }
    info.insert("LSUIElement".into(), plist::Value::Boolean(true));
    plist::Value::Dictionary(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mac_bundle_has_microphone_purpose_and_stable_identity() {
        let info = bundle_info();
        let dict = info.as_dictionary().unwrap();
        assert_eq!(dict["CFBundleIdentifier"].as_string(), Some(crate::LABEL));
        assert!(dict.contains_key("NSMicrophoneUsageDescription"));
        assert_eq!(dict["LSUIElement"].as_boolean(), Some(true));
    }
    #[test]
    fn atomic_copy_and_replace() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("voco-copy-test-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let source = dir.join("source");
        let destination = dir.join("nested/destination");
        fs::write(&source, "one")?;
        copy_atomic(&source, &destination)?;
        assert_eq!(fs::read(&destination)?, b"one");
        fs::write(&source, "replacement")?;
        copy_atomic(&source, &destination)?;
        assert_eq!(fs::read(&destination)?, b"replacement");
        fs::remove_dir_all(dir)?;
        Ok(())
    }
}
