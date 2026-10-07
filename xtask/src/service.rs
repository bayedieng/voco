//! service-manager supplies installation/lifecycle; explicit templates handle quoting and GUI scope.
use crate::{LABEL, Result, deploy::Layout};
use service_manager::{
    ServiceInstallCtx, ServiceLevel, ServiceManager, ServiceManagerKind, ServiceStartCtx,
    ServiceStatus, ServiceStatusCtx, ServiceStopCtx,
};
use std::{ffi::OsString, path::PathBuf, process::Command};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Linux,
    Macos,
}
impl Platform {
    pub fn native() -> Result<Self> {
        if cfg!(target_os = "linux") {
            Ok(Self::Linux)
        } else if cfg!(target_os = "macos") {
            Ok(Self::Macos)
        } else {
            Err(
                "desktop service deployment currently supports Linux/systemd and macOS/launchd"
                    .into(),
            )
        }
    }
}

pub struct InstallOptions {
    pub device: Option<String>,
    pub model_idle_secs: u64,
    pub max_recording_secs: u64,
    pub keep_model_loaded: bool,
    pub print_only: bool,
    pub hotkey: String,
    pub external_hotkey: bool,
    pub no_cues: bool,
    pub no_start: bool,
}

pub fn manager(platform: Platform) -> Result<Box<dyn ServiceManager>> {
    let mut manager = <dyn ServiceManager>::target(match platform {
        Platform::Linux => ServiceManagerKind::Systemd,
        Platform::Macos => ServiceManagerKind::Launchd,
    });
    manager.set_level(ServiceLevel::User)?;
    if !manager.available()? {
        return Err("native user-session service manager is unavailable".into());
    }
    Ok(manager)
}

pub fn install_context(
    platform: Platform,
    layout: &Layout,
    options: &InstallOptions,
) -> Result<ServiceInstallCtx> {
    let mut args: Vec<OsString> = vec![
        "--daemon".into(),
        "--model-dir".into(),
        layout.model_dir.clone().into(),
        "--model-idle-secs".into(),
        options.model_idle_secs.to_string().into(),
        "--max-recording-secs".into(),
        options.max_recording_secs.to_string().into(),
        "--hotkey".into(),
        options.hotkey.clone().into(),
        "--control-socket".into(),
        layout.root.join("control.sock").into(),
    ];
    if let Some(device) = &options.device {
        args.extend(["--device".into(), device.into()]);
    }
    for (enabled, flag) in [
        (options.keep_model_loaded, "--keep-model-loaded"),
        (options.print_only, "--print-only"),
        (options.external_hotkey, "--external-hotkey"),
        (options.no_cues, "--no-cues"),
    ] {
        if enabled {
            args.push(flag.into());
        }
    }
    let mut ctx = ServiceInstallCtx {
        label: LABEL.parse()?,
        program: layout.program.clone(),
        args,
        contents: None,
        username: None,
        working_directory: Some(layout.root.clone()),
        environment: None,
        autostart: !options.no_start,
        restart_policy: service_manager::RestartPolicy::OnFailure {
            delay_secs: Some(5),
            max_retries: None,
            reset_after_secs: None,
        },
    };
    ctx.contents = Some(match platform {
        Platform::Linux => linux_unit(&ctx)?,
        Platform::Macos => launch_agent(&ctx, layout, options.no_start)?,
    });
    Ok(ctx)
}

fn quote(value: &str, command: bool) -> Result<String> {
    if value.contains(['\0', '\n', '\r']) {
        return Err("service arguments cannot contain NUL/newlines".into());
    }
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('%', "%%");
    Ok(format!(
        "\"{}\"",
        if command {
            escaped.replace('$', "$$")
        } else {
            escaped
        }
    ))
}

fn linux_unit(ctx: &ServiceInstallCtx) -> Result<String> {
    let command: Vec<_> = ctx
        .cmd_iter()
        .map(|arg| quote(arg.to_str().ok_or("non-UTF8 service argument")?, true))
        .collect::<Result<_>>()?;
    // Unlike ExecStart, WorkingDirectory is a single raw path: quotes become literal characters.
    let directory = ctx
        .working_directory
        .as_ref()
        .ok_or("missing working directory")?
        .to_str()
        .ok_or("non-UTF8 directory")?;
    if directory.contains(['\0', '\n', '\r']) {
        return Err("directory cannot contain NUL/newlines".into());
    }
    let directory = directory.replace('%', "%%");
    Ok(format!(
        "[Unit]\nDescription=Voco toggle dictation\nAfter=graphical-session.target\nPartOf=graphical-session.target\nStartLimitIntervalSec=120\nStartLimitBurst=5\n\n[Service]\nType=simple\nExecStart={}\nWorkingDirectory={directory}\nRestart=on-failure\nRestartSec=5\nTimeoutStopSec=30\nUMask=0077\n\n[Install]\nWantedBy=graphical-session.target\n",
        command.join(" ")
    ))
}

fn launch_agent(ctx: &ServiceInstallCtx, layout: &Layout, disabled: bool) -> Result<String> {
    use plist::{Dictionary, Value};
    let mut dict = Dictionary::new();
    dict.insert("Label".into(), Value::String(LABEL.into()));
    dict.insert(
        "ProgramArguments".into(),
        Value::Array(
            ctx.cmd_iter()
                .map(|arg| {
                    arg.to_str()
                        .map(|v| Value::String(v.into()))
                        .ok_or("non-UTF8 launchd argument")
                })
                .collect::<std::result::Result<_, _>>()?,
        ),
    );
    dict.insert(
        "WorkingDirectory".into(),
        Value::String(layout.root.to_str().ok_or("non-UTF8 path")?.into()),
    );
    dict.insert(
        "LimitLoadToSessionType".into(),
        Value::String("Aqua".into()),
    );
    dict.insert("RunAtLoad".into(), Value::Boolean(true));
    dict.insert("Disabled".into(), Value::Boolean(disabled));
    let mut keep_alive = Dictionary::new();
    keep_alive.insert("SuccessfulExit".into(), Value::Boolean(false));
    dict.insert("KeepAlive".into(), Value::Dictionary(keep_alive));
    dict.insert("ThrottleInterval".into(), Value::Integer(5.into()));
    dict.insert("ExitTimeOut".into(), Value::Integer(30.into()));
    dict.insert(
        "StandardOutPath".into(),
        Value::String(
            layout
                .logs
                .join("stdout.log")
                .to_string_lossy()
                .into_owned(),
        ),
    );
    dict.insert(
        "StandardErrorPath".into(),
        Value::String(
            layout
                .logs
                .join("stderr.log")
                .to_string_lossy()
                .into_owned(),
        ),
    );
    let mut bytes = Vec::new();
    Value::Dictionary(dict).to_writer_xml(&mut bytes)?;
    Ok(String::from_utf8(bytes)?)
}

pub fn import_session_environment(platform: Platform) -> Result<()> {
    if platform == Platform::Linux {
        let keys: Vec<_> = [
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "XAUTHORITY",
            "XDG_RUNTIME_DIR",
            "DBUS_SESSION_BUS_ADDRESS",
            "XDG_SESSION_TYPE",
        ]
        .into_iter()
        .filter(|key| std::env::var_os(key).is_some())
        .collect();
        if !keys.is_empty() {
            let status = Command::new("systemctl")
                .args(["--user", "import-environment"])
                .args(keys)
                .status()?;
            if !status.success() {
                return Err("cannot import graphical-session environment into user systemd".into());
            }
        }
    }
    Ok(())
}

pub fn linux_unit_name() -> Result<String> {
    let label: service_manager::ServiceLabel = LABEL.parse()?;
    Ok(format!("{}.service", label.to_script_name()))
}

/// Use the same directory/name as service-manager, including XDG_CONFIG_HOME overrides.
pub fn registration_file(platform: Platform) -> Result<PathBuf> {
    match platform {
        Platform::Linux => Ok(service_manager::systemd_user_dir_path()?.join(linux_unit_name()?)),
        Platform::Macos => agent_file(),
    }
}

pub fn require_installed(platform: Platform) -> Result<PathBuf> {
    let file = registration_file(platform)?;
    if !file.is_file() {
        return Err(format!(
            "Voco service is not installed at {}. Run `cargo xtask service install` first; `--dry-run` does not install anything.",
            file.display()
        ).into());
    }
    Ok(file)
}

fn reload_user_manager(platform: Platform) -> Result<()> {
    if platform == Platform::Linux
        && !Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .status()?
            .success()
    {
        return Err("cannot reload systemd user-service definitions".into());
    }
    Ok(())
}

fn agent_file() -> Result<PathBuf> {
    let base = directories::BaseDirs::new().ok_or("no home directory")?;
    Ok(base
        .home_dir()
        .join(format!("Library/LaunchAgents/{LABEL}.plist")))
}

pub fn start(platform: Platform, manager: &dyn ServiceManager, _: &Layout) -> Result<()> {
    if platform == Platform::Macos
        && manager.status(ServiceStatusCtx {
            label: LABEL.parse()?,
        })? == ServiceStatus::NotInstalled
    {
        let status = Command::new("launchctl")
            .arg("load")
            .arg(agent_file()?)
            .status()?;
        if !status.success() {
            return Err("cannot reload LaunchAgent".into());
        }
    }
    manager.start(ServiceStartCtx {
        label: LABEL.parse()?,
    })?;
    Ok(())
}

pub fn stop(platform: Platform, manager: &dyn ServiceManager, _: &Layout) -> Result<()> {
    manager.stop(ServiceStopCtx {
        label: LABEL.parse()?,
    })?;
    if platform == Platform::Macos {
        // Unload as well: KeepAlive must not restart an explicitly stopped agent.
        let status = Command::new("launchctl")
            .arg("unload")
            .arg(agent_file()?)
            .status()?;
        if !status.success() {
            return Err("cannot unload LaunchAgent".into());
        }
    }
    Ok(())
}

fn user_id() -> Result<u32> {
    let uid = Command::new("id").arg("-u").output()?;
    if !uid.status.success() {
        return Err("cannot determine user id".into());
    }
    let uid = String::from_utf8(uid.stdout)?;
    Ok(uid.trim().parse()?)
}

pub fn require_desktop_user() -> Result<()> {
    if user_id()? == 0 {
        return Err("Root is neither required nor supported for desktop deployment. Re-run as your logged-in user without sudo; Voco never silently elevates privileges.".into());
    }
    Ok(())
}

fn mac_domain_label() -> Result<String> {
    Ok(format!("gui/{}/{LABEL}", user_id()?))
}

/// Permanent policy changes; the crate has no enable/disable trait methods.
pub fn load(platform: Platform, manager: &dyn ServiceManager, layout: &Layout) -> Result<()> {
    let file = require_installed(platform)?;
    reload_user_manager(platform)?;
    let status = match platform {
        Platform::Linux => Command::new("systemctl")
            .args(["--user", "enable"])
            .arg(&file)
            .status()?,
        Platform::Macos => Command::new("launchctl")
            .arg("enable")
            .arg(mac_domain_label()?)
            .status()?,
    };
    if !status.success() {
        return Err("cannot enable login autostart".into());
    }
    start(platform, manager, layout)
}

pub fn unload(platform: Platform, manager: &dyn ServiceManager, layout: &Layout) -> Result<()> {
    reload_user_manager(platform)?;
    if manager.status(ServiceStatusCtx {
        label: LABEL.parse()?,
    })? == ServiceStatus::Running
    {
        stop(platform, manager, layout)?;
    }
    let status = match platform {
        Platform::Linux => Command::new("systemctl")
            .args(["--user", "disable"])
            .arg(registration_file(platform)?)
            .status()?,
        Platform::Macos => Command::new("launchctl")
            .arg("disable")
            .arg(mac_domain_label()?)
            .status()?,
    };
    if !status.success() {
        return Err("cannot disable login autostart".into());
    }
    Ok(())
}

pub fn log_hint(platform: Platform, layout: &Layout) -> Result<String> {
    Ok(match platform {
        Platform::Linux => format!("journalctl --user -u {}", linux_unit_name()?),
        Platform::Macos => layout.logs.display().to_string(),
    })
}

pub fn permission_guidance(platform: Platform, layout: &Layout) {
    println!(
        "Toggle with Ctrl+Alt+Z (Control+Option+Z on macOS), or {} --toggle",
        layout.program.display()
    );
    if platform == Platform::Macos {
        println!(
            "Grant Microphone and Accessibility permissions to {} in System Settings. Ad-hoc signing may require reauthorization after updates; use a stable Developer ID signature for production.",
            layout.bundle.display()
        );
    } else {
        println!(
            "Wayland needs a GlobalShortcuts portal. If unavailable, bind {} --toggle in your compositor and deploy with --external-hotkey.",
            layout.program.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    fn options() -> InstallOptions {
        InstallOptions {
            device: Some("Mic with spaces".into()),
            model_idle_secs: 60,
            max_recording_secs: 60,
            keep_model_loaded: false,
            print_only: false,
            hotkey: "Ctrl+Alt+Z".into(),
            external_hotkey: false,
            no_cues: false,
            no_start: false,
        }
    }
    #[test]
    fn linux_quotes_paths_and_arguments_without_shell() -> Result<()> {
        let layout = Layout::at(Path::new("/home/example/a % $dir").into(), Platform::Linux);
        let ctx = install_context(Platform::Linux, &layout, &options())?;
        let text = ctx.contents.unwrap();
        assert!(text.contains("\"/home/example/a %% $$dir/bin/vocod\""));
        assert!(text.contains("\"Mic with spaces\""));
        assert!(text.contains("WantedBy=graphical-session.target"));
        assert!(text.contains("WorkingDirectory=/home/example/a %% $dir\n"));
        assert!(quote("bad\npath", true).is_err());
        Ok(())
    }
    #[test]
    fn mac_is_a_user_gui_agent_with_quoted_plist_paths() -> Result<()> {
        let layout = Layout::at(
            Path::new("/Users/example/Library/Application Support/voco").into(),
            Platform::Macos,
        );
        let ctx = install_context(Platform::Macos, &layout, &options())?;
        let value = plist::Value::from_reader_xml(ctx.contents.unwrap().as_bytes())?;
        let dict = value.as_dictionary().unwrap();
        assert_eq!(dict["LimitLoadToSessionType"].as_string(), Some("Aqua"));
        assert_eq!(dict["Disabled"].as_boolean(), Some(false));
        assert!(
            dict["ProgramArguments"].as_array().unwrap()[0]
                .as_string()
                .unwrap()
                .contains("Application Support")
        );
        assert!(!dict.contains_key("UserName"));
        Ok(())
    }
}
