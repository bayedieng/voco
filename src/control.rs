//! Private per-user IPC: compositor shortcuts can run `vocod --toggle` without privileges.
use crate::{Result, wake::Wake};
use std::{
    io::{BufRead, BufReader, Write},
    os::unix::{
        fs::{DirBuilderExt, FileTypeExt, MetadataExt, PermissionsExt},
        net::{UnixListener, UnixStream},
    },
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, Ordering},
        mpsc::SyncSender,
    },
    time::Duration,
};

pub const IDLE: u8 = 0;
pub const RECORDING: u8 = 1;
pub const TRANSCRIBING: u8 = 2;

pub struct Control {
    pub stop: AtomicBool,
    pub state: AtomicU8,
    pub keys_down: AtomicBool,
    pub warmup: AtomicBool,
    pub recorder: Arc<Wake>,
    pub inference: Arc<Wake>,
    pub ui: Arc<Wake>,
    pub toggles: SyncSender<()>,
}
impl Control {
    pub fn toggle(&self) {
        if self.toggles.try_send(()).is_ok() {
            self.recorder.notify();
        }
    }
    pub fn key(&self, down: bool) {
        let previous = self.keys_down.swap(down, Ordering::AcqRel);
        if down && !previous {
            self.toggle();
        }
        if !down {
            self.inference.notify();
        }
    }
    pub fn shutdown(&self) {
        self.stop.store(true, Ordering::Release);
        self.keys_down.store(false, Ordering::Release);
        self.recorder.notify();
        self.inference.notify();
        self.ui.notify();
        crate::hotkey::wake_main();
    }
    pub fn status(&self) -> &'static str {
        match self.state.load(Ordering::Acquire) {
            RECORDING => "recording",
            TRANSCRIBING => "transcribing",
            _ => "idle",
        }
    }
}

pub fn default_socket() -> Result<PathBuf> {
    Ok(directories::BaseDirs::new()
        .ok_or("no user directories")?
        .data_local_dir()
        .join("voco/control.sock"))
}

pub fn request(path: &Path, action: &str) -> Result<String> {
    let mut stream = UnixStream::connect(path)
        .map_err(|error| format!("daemon not reachable at {}: {error}", path.display()))?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    writeln!(stream, "{action}")?;
    let mut reply = String::new();
    BufReader::new(stream).read_line(&mut reply)?;
    Ok(reply)
}

pub struct SocketGuard {
    path: PathBuf,
    device: u64,
    inode: u64,
}
impl Drop for SocketGuard {
    fn drop(&mut self) {
        // A fast restart may have replaced our stale socket before this scope exits.
        if let Ok(metadata) = std::fs::symlink_metadata(&self.path)
            && metadata.file_type().is_socket()
            && metadata.dev() == self.device
            && metadata.ino() == self.inode
        {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

pub fn bind(path: &Path) -> Result<(UnixListener, SocketGuard)> {
    let parent = path.parent().ok_or("socket needs a parent directory")?;
    if !parent.exists() {
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(parent)?;
    }
    if std::fs::metadata(parent)?.permissions().mode() & 0o077 != 0 {
        return Err(format!(
            "control socket directory must be private (chmod 700): {}",
            parent.display()
        )
        .into());
    }
    if path.exists() {
        if UnixStream::connect(path).is_ok() {
            return Err("another vocod daemon is already running".into());
        }
        if !std::fs::symlink_metadata(path)?.file_type().is_socket() {
            return Err("refusing to remove a non-socket control file".into());
        }
        std::fs::remove_file(path)?;
    }
    let listener = UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    let metadata = std::fs::symlink_metadata(path)?;
    Ok((
        listener,
        SocketGuard {
            path: path.to_owned(),
            device: metadata.dev(),
            inode: metadata.ino(),
        },
    ))
}

pub fn serve(listener: UnixListener, control: Arc<Control>) -> Result<()> {
    for stream in listener.incoming() {
        if control.stop.load(Ordering::Acquire) {
            break;
        }
        let mut stream = stream?;
        stream.set_read_timeout(Some(Duration::from_secs(1)))?;
        stream.set_write_timeout(Some(Duration::from_secs(1)))?;
        let mut line = String::new();
        // Bound input and time: a stalled local client must not hold the control server forever.
        use std::io::Read;
        if BufReader::new((&mut stream).take(64))
            .read_line(&mut line)
            .is_err()
        {
            continue;
        }
        let reply = match line.trim() {
            "toggle" => {
                control.toggle();
                "ok"
            }
            "status" => control.status(),
            "quit" => {
                control.shutdown();
                "stopping"
            }
            _ => "error: unknown command",
        };
        let _ = writeln!(stream, "{reply}");
        if control.stop.load(Ordering::Acquire) {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hotkey_repeat_only_toggles_once_until_release() {
        let (tx, rx) = std::sync::mpsc::sync_channel(4);
        let c = Control {
            stop: AtomicBool::new(false),
            state: AtomicU8::new(IDLE),
            keys_down: AtomicBool::new(false),
            warmup: AtomicBool::new(false),
            recorder: Arc::default(),
            inference: Arc::default(),
            ui: Arc::default(),
            toggles: tx,
        };
        c.key(true);
        c.key(true);
        c.key(true);
        assert_eq!(rx.try_iter().count(), 1);
        c.key(false);
        c.key(true);
        assert_eq!(rx.try_iter().count(), 1);
    }

    #[test]
    fn private_socket_roundtrip_duplicate_refusal_and_shutdown() -> Result<()> {
        let root = std::env::temp_dir().join(format!("voco-control-{}", std::process::id()));
        let path = root.join("control.sock");
        let (listener, guard) = bind(&path)?;
        assert_eq!(
            std::fs::metadata(&path)?.permissions().mode() & 0o777,
            0o600
        );
        let (tx, rx) = std::sync::mpsc::sync_channel(4);
        let control = Arc::new(Control {
            stop: AtomicBool::new(false),
            state: AtomicU8::new(IDLE),
            keys_down: AtomicBool::new(false),
            warmup: AtomicBool::new(false),
            recorder: Arc::default(),
            inference: Arc::default(),
            ui: Arc::default(),
            toggles: tx,
        });
        let worker = std::thread::spawn(move || serve(listener, control));
        assert!(bind(&path).is_err());
        assert_eq!(request(&path, "status")?.trim(), "idle");
        assert_eq!(request(&path, "toggle")?.trim(), "ok");
        rx.recv_timeout(Duration::from_secs(1))?;
        assert_eq!(request(&path, "quit")?.trim(), "stopping");
        worker.join().unwrap()?;
        drop(guard);
        assert!(!path.exists());
        std::fs::remove_dir(root)?;
        Ok(())
    }
}
