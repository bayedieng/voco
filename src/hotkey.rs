//! Native X11/macOS shortcuts; Wayland uses the consent-based GlobalShortcuts portal.
use crate::{Result, control::Control};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState, hotkey::HotKey};
use std::sync::{Arc, atomic::Ordering};

#[cfg(target_os = "macos")]
#[path = "hotkey_macos.rs"]
mod macos;

pub struct Registration {
    _native: Option<GlobalHotKeyManager>,
}

pub fn register(shortcut: &str, external: bool, control: Arc<Control>) -> Result<Registration> {
    if external {
        return Ok(Registration { _native: None });
    }
    let hotkey: HotKey = shortcut.parse()?;
    #[cfg(target_os = "linux")]
    if std::env::var_os("WAYLAND_DISPLAY").is_some()
        || std::env::var("XDG_SESSION_TYPE").is_ok_and(|s| s == "wayland")
    {
        let preferred = shortcut.to_uppercase();
        std::thread::Builder::new().name("shortcut-portal".into()).spawn(move || {
            let result = portal(&preferred, control);
            if let Err(error) = result { eprintln!("GlobalShortcuts portal unavailable: {error}. Bind 'vocod --toggle' in the compositor, or use --external-hotkey."); }
        })?;
        return Ok(Registration { _native: None });
    }
    let manager = GlobalHotKeyManager::new()?;
    manager.register(hotkey)?;
    #[cfg(target_os = "macos")]
    macos::prepare()?;
    GlobalHotKeyEvent::set_event_handler(Some(move |event: GlobalHotKeyEvent| {
        if event.id == hotkey.id() {
            control.key(event.state == HotKeyState::Pressed);
        }
    }));
    Ok(Registration {
        _native: Some(manager),
    })
}

#[cfg(target_os = "linux")]
fn portal(preferred: &str, control: Arc<Control>) -> Result<()> {
    use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut};
    use futures_util::StreamExt;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        if let Err(error) = ashpd::register_host_app("com.voco.vocod".parse()?).await {
            eprintln!("Portal app identity registration unavailable: {error}");
        }
        let proxy = GlobalShortcuts::new().await?;
        let session = proxy.create_session(Default::default()).await?;
        let mut activated = proxy.receive_activated().await?;
        let mut deactivated = proxy.receive_deactivated().await?;
        let shortcuts = [NewShortcut::new("toggle", "Start/stop Voco dictation").preferred_trigger(Some(preferred))];
        let response = proxy.bind_shortcuts(&session, &shortcuts, None, Default::default()).await?.response()?;
        if let Some(shortcut) = response.shortcuts().first() { eprintln!("Dictation shortcut: {}", shortcut.trigger_description()); }
        loop {
            // No timer polling: block on actual shortcut events. Process exit closes the session.
            tokio::select! {
                event = activated.next() => match event { Some(e) if e.shortcut_id() == "toggle" => control.key(true), Some(_) => {}, None => break },
                event = deactivated.next() => match event { Some(e) if e.shortcut_id() == "toggle" => control.key(false), Some(_) => {}, None => break },
            }
        }
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })
}

impl Registration {
    pub fn run_main_loop(&self, control: &Control) -> Result<()> {
        run_main_loop(control, self._native.is_some())
    }
}

fn run_main_loop(control: &Control, _native: bool) -> Result<()> {
    #[cfg(target_os = "macos")]
    if _native {
        return macos::run(control);
    }
    while !control.stop.load(Ordering::Acquire) {
        control.ui.wait(None);
    }
    Ok(())
}

pub fn wake_main() {
    #[cfg(target_os = "macos")]
    macos::wake();
}
