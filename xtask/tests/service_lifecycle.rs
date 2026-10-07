//! Exercise the actual service-manager installer, but intercept all native commands.
//! No real service, microphone or privilege elevation is involved.
#![cfg(target_os = "linux")]
use std::{
    fs,
    os::unix::fs::{DirBuilderExt, PermissionsExt},
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    root: PathBuf,
}
impl Fixture {
    fn new() -> Result<Self> {
        let root = std::env::temp_dir().join(format!(
            "voco-lifecycle-{}-{} with spaces",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&root)?;
        let fixture = Self { root };
        fs::create_dir_all(fixture.root.join("bin"))?;
        fixture.script("id", "#!/bin/sh\nprintf '1000\\n'\n")?;
        fixture.script(
            "cargo",
            "#!/bin/sh\nprintf '%s\\n' \"$VOCO_TEST_METADATA\"\n",
        )?;
        fixture.script(
            "systemctl",
            r#"#!/bin/sh
printf '%s\n' "$*" >> "$VOCO_TEST_LOG"
[ "$1" = '--user' ] || exit 90
case "$2" in
    daemon-reload) : > "$VOCO_TEST_RELOADED" ;;
    status)
        [ -f "$VOCO_TEST_UNIT" ] || exit 4
        [ -f "$VOCO_TEST_RUNNING" ] && exit 0
        exit 3 ;;
    enable)
        [ "$3" = "$VOCO_TEST_UNIT" ] && [ -f "$3" ] && [ -f "$VOCO_TEST_RELOADED" ] || exit 91
        : > "$VOCO_TEST_ENABLED" ;;
    disable)
        [ "$3" = "$VOCO_TEST_UNIT" ] && [ -f "$3" ] || exit 92
        rm -f "$VOCO_TEST_ENABLED" ;;
    start)
        [ "$3" = 'voco-vocod' ] && [ -f "$VOCO_TEST_UNIT" ] || exit 93
        : > "$VOCO_TEST_RUNNING" ;;
    stop)
        [ "$3" = 'voco-vocod' ] || exit 94
        rm -f "$VOCO_TEST_RUNNING" ;;
    *) exit 95 ;;
esac
"#,
        )?;
        Ok(fixture)
    }
    fn script(&self, name: &str, contents: &str) -> Result<()> {
        let path = self.root.join("bin").join(name);
        fs::write(&path, contents)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
        Ok(())
    }
    fn unit(&self) -> PathBuf {
        self.root.join("config/systemd/user/voco-vocod.service")
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command
            .arg("service")
            .env("HOME", self.root.join("home"))
            .env("XDG_CONFIG_HOME", self.root.join("config"))
            .env("XDG_DATA_HOME", self.root.join("data"))
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            )
            .env("CARGO", self.root.join("bin/cargo"))
            .env(
                "VOCO_TEST_METADATA",
                serde_json::json!({"target_directory": self.root.join("target")}).to_string(),
            )
            .env("VOCO_TEST_UNIT", self.unit())
            .env("VOCO_TEST_LOG", self.root.join("commands"))
            .env("VOCO_TEST_RELOADED", self.root.join("reloaded"))
            .env("VOCO_TEST_ENABLED", self.root.join("enabled"))
            .env("VOCO_TEST_RUNNING", self.root.join("running"));
        for key in [
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "XAUTHORITY",
            "XDG_RUNTIME_DIR",
            "DBUS_SESSION_BUS_ADDRESS",
            "XDG_SESSION_TYPE",
        ] {
            command.env_remove(key);
        }
        command
    }
    fn success(&self, action: &str) -> Result<Output> {
        let output = self.command().arg(action).output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(output)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn load_without_install_reports_action_before_native_commands() -> Result<()> {
    let fixture = Fixture::new()?;
    let output = fixture.command().arg("load").output()?;
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr)?;
    assert!(error.contains("cargo xtask service install"), "{error}");
    assert!(error.contains("--dry-run"), "{error}");
    assert!(!fixture.root.join("commands").exists());
    assert!(!fixture.unit().exists());
    Ok(())
}

#[test]
fn no_start_install_then_load_unload_uses_registered_path_and_reload() -> Result<()> {
    let fixture = Fixture::new()?;
    let binary = fixture.root.join("target/release/vocod");
    fs::create_dir_all(binary.parent().unwrap())?;
    fs::write(binary, "fixture executable; never run")?;
    let models = fixture.root.join("source-models");
    fs::create_dir_all(&models)?;
    for file in [
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ] {
        fs::write(models.join(file), "fixture model; never loaded")?;
    }
    let output = fixture
        .command()
        .args(["install", "--skip-build", "--no-start", "--model-dir"])
        .arg(models)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(fixture.unit().is_file());
    assert!(!fixture.root.join("running").exists());
    assert!(!fixture.root.join("enabled").exists());
    assert!(fs::read_to_string(fixture.root.join("commands"))?.contains("--user daemon-reload"));

    fs::write(fixture.root.join("commands"), "")?;
    fs::remove_file(fixture.root.join("reloaded"))?;
    fixture.success("load")?;
    let commands = fs::read_to_string(fixture.root.join("commands"))?;
    assert_eq!(
        commands.lines().collect::<Vec<_>>(),
        vec![
            "--user daemon-reload".to_owned(),
            format!("--user enable {}", fixture.unit().display()),
            "--user start voco-vocod".to_owned(),
        ]
    );
    assert!(fixture.root.join("running").exists());
    assert!(fixture.root.join("enabled").exists());

    fixture.success("unload")?;
    assert!(!fixture.root.join("running").exists());
    assert!(!fixture.root.join("enabled").exists());
    assert!(fixture.unit().is_file()); // retained for future load
    fixture.success("load")?;
    assert!(fixture.root.join("running").exists());
    Ok(())
}
