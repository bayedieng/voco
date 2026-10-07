mod deploy;
mod service;

use clap::{Parser, Subcommand};
use service_manager::{ServiceStatusCtx, ServiceUninstallCtx};
use std::{
    error::Error,
    path::{Path, PathBuf},
    process::Command,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const LABEL: &str = "com.voco.vocod";
const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

#[derive(Parser)]
#[command(about = "Build and deploy Voco's per-user toggle dictation service")]
struct Args {
    #[command(subcommand)]
    command: Task,
}
#[derive(Subcommand)]
enum Task {
    Service {
        #[command(subcommand)]
        command: ServiceTask,
    },
}
#[derive(Subcommand)]
enum ServiceTask {
    /// Build, deploy, register for login and start the daemon (never use sudo)
    Install {
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        skip_build: bool,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long)]
        device: Option<String>,
        #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u64).range(1..=86400))]
        model_idle_secs: u64,
        #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u64).range(1..=120))]
        max_recording_secs: u64,
        #[arg(long)]
        keep_model_loaded: bool,
        #[arg(long)]
        print_only: bool,
        #[arg(long, default_value = "Ctrl+Alt+Z")]
        hotkey: String,
        #[arg(long)]
        external_hotkey: bool,
        #[arg(long)]
        no_cues: bool,
        /// Deploy without starting, e.g. to authorize permissions in foreground first
        #[arg(long)]
        no_start: bool,
    },
    /// Enable login autostart and start the deployed daemon permanently
    Load,
    /// Stop and disable login autostart, retaining deployment and service files
    Unload,
    /// Start this session without changing autostart policy
    Start,
    /// Stop this session; login autostart remains enabled
    Stop,
    Restart,
    Status,
    /// Toggle recording in the existing daemon
    Toggle,
    /// Run the deployed executable in the terminal to authorize desktop/mic permissions
    Run,
    /// Remove service registration; retain deployed binary and model files
    Uninstall,
}

fn main() -> Result<()> {
    let Task::Service { command } = Args::parse().command;
    let platform = service::Platform::native()?;
    if !matches!(
        &command,
        ServiceTask::Install { dry_run: true, .. } | ServiceTask::Status
    ) {
        service::require_desktop_user()?;
    }
    let layout = deploy::Layout::for_user(platform)?;
    match command {
        ServiceTask::Install {
            dry_run,
            skip_build,
            model_dir,
            device,
            model_idle_secs,
            max_recording_secs,
            keep_model_loaded,
            print_only,
            hotkey,
            external_hotkey,
            no_cues,
            no_start,
        } => {
            let options = service::InstallOptions {
                device,
                model_idle_secs,
                max_recording_secs,
                keep_model_loaded,
                print_only,
                hotkey,
                external_hotkey,
                no_cues,
                no_start,
            };
            let context = service::install_context(platform, &layout, &options)?;
            if dry_run {
                println!(
                    "Deploy to: {}\nProgram: {}\nService definition:\n{}",
                    layout.root.display(),
                    context.program.display(),
                    context.contents.as_deref().unwrap_or_default()
                );
                return Ok(());
            }
            let root = Path::new(ROOT).canonicalize()?;
            let binary = if skip_build {
                deploy::existing_release(&root)?
            } else {
                deploy::build(&root)?
            };
            let source = model_dir
                .unwrap_or_else(|| root.join("models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"));
            let manager = service::manager(platform)?;
            if manager.status(ServiceStatusCtx {
                label: LABEL.parse()?,
            })? == service_manager::ServiceStatus::Running
            {
                service::stop(platform, manager.as_ref(), &layout)?;
            }
            deploy::copy_release(&binary, &source, &layout, platform)?;
            service::import_session_environment(platform)?;
            manager.install(context)?;
            if no_start {
                service::unload(platform, manager.as_ref(), &layout)?;
            } else {
                service::load(platform, manager.as_ref(), &layout)?;
            }
            println!(
                "Installed {LABEL} as a user-session service. Files: {}",
                layout.root.display()
            );
            service::permission_guidance(platform, &layout);
        }
        ServiceTask::Load => {
            service::require_installed(platform)?;
            let manager = service::manager(platform)?;
            service::import_session_environment(platform)?;
            service::load(platform, manager.as_ref(), &layout)?;
        }
        ServiceTask::Unload => {
            service::unload(platform, service::manager(platform)?.as_ref(), &layout)?
        }
        ServiceTask::Start => {
            let manager = service::manager(platform)?;
            service::import_session_environment(platform)?;
            service::start(platform, manager.as_ref(), &layout)?;
        }
        ServiceTask::Stop => {
            service::stop(platform, service::manager(platform)?.as_ref(), &layout)?
        }
        ServiceTask::Restart => {
            let manager = service::manager(platform)?;
            service::stop(platform, manager.as_ref(), &layout)?;
            service::import_session_environment(platform)?;
            service::start(platform, manager.as_ref(), &layout)?;
        }
        ServiceTask::Status => {
            let manager = service::manager(platform)?;
            println!(
                "{LABEL}: {:?}",
                manager.status(ServiceStatusCtx {
                    label: LABEL.parse()?
                })?
            );
            println!(
                "Files: {}\nLogs: {}",
                layout.root.display(),
                service::log_hint(platform, &layout)?
            );
            if layout.root.join("control.sock").exists() {
                let _ = Command::new(&layout.program)
                    .arg("--daemon-status")
                    .status();
            }
        }
        ServiceTask::Toggle => {
            if !Command::new(&layout.program)
                .arg("--toggle")
                .status()?
                .success()
            {
                return Err("daemon toggle failed".into());
            }
        }
        ServiceTask::Run => {
            if !Command::new(&layout.program)
                .args(["--daemon", "--model-dir"])
                .arg(&layout.model_dir)
                .status()?
                .success()
            {
                return Err("foreground permission/recording run failed".into());
            }
        }
        ServiceTask::Uninstall => {
            let manager = service::manager(platform)?;
            if manager.status(ServiceStatusCtx {
                label: LABEL.parse()?,
            })? == service_manager::ServiceStatus::Running
            {
                service::stop(platform, manager.as_ref(), &layout)?;
            }
            manager.uninstall(ServiceUninstallCtx {
                label: LABEL.parse()?,
            })?;
            println!(
                "Service removed. Deployed files retained at {}",
                layout.root.display()
            );
        }
    }
    Ok(())
}
