use std::env;
use std::error::Error;
use std::path::Path;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("xtask: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, Box<dyn Error>> {
    let mut args = env::args_os().skip(1);

    match args.next().as_deref().and_then(|arg| arg.to_str()) {
        Some("build") => {
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .ok_or("cannot locate workspace root")?;
            let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
            let status = Command::new(cargo)
                .current_dir(root)
                .args(["build", "--workspace", "--exclude", "xtask"])
                .args(args)
                .status()?;

            Ok(if status.success() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        None | Some("help" | "-h" | "--help") => {
            println!("Usage: cargo xtask build [cargo build options]");
            println!();
            println!("Builds voco-lib and vocod. Extra arguments are forwarded to cargo build.");
            Ok(ExitCode::SUCCESS)
        }
        _ => Err("unknown task; use cargo xtask --help".into()),
    }
}
