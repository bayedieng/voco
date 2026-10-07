//! XWayland's native keymap + Shift, avoiding Enigo's dynamic keysym remapping.
//! KWin/Mutter can forward XTest keys but do not forward X11 keymap changes to
//! native Wayland clients. Read XKB here; all injection still goes through Enigo.
use crate::Result;
use enigo::{Direction, Keyboard};
use std::{collections::HashMap, thread, time::Duration};
use wayland_client::{Connection, Dispatch, QueueHandle, protocol::wl_registry};
use x11rb::{
    protocol::{
        xkb::{ConnectionExt as _, ID, KeySymMap, KeyType, MapPart},
        xproto::ConnectionExt as _,
    },
    rust_connection::RustConnection,
};

pub struct XwaylandKeys {
    connection: RustConnection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stroke {
    code: u16,
    shift: bool,
}

// Enigo's raw XTest events have no built-in pacing. An X11 roundtrip only
// acknowledges XWayland, not KWin/Mutter or the receiving Wayland application.
// Let modifier updates settle and keep key-down/up events separated so the
// receiver does not apply Shift to subsequent lowercase letters.
const MODIFIER_SETTLE: Duration = Duration::from_millis(20);
const KEY_INTERVAL: Duration = Duration::from_millis(5);

impl XwaylandKeys {
    pub fn detect() -> Result<Option<Self>> {
        let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some()
            || std::env::var("XDG_SESSION_TYPE").is_ok_and(|s| s == "wayland");
        if !wayland || native_wayland_keyboard() {
            return Ok(None);
        }
        let (connection, _) = x11rb::connect(None)?;
        if !connection.xkb_use_extension(1, 0)?.reply()?.supported {
            return Err("XWayland input requires the XKB extension".into());
        }
        eprintln!("Dictation output: Enigo/XWayland with native keymap + Shift");
        Ok(Some(Self { connection }))
    }

    pub fn text(&self, keyboard: &mut impl Keyboard, text: &str) -> Result<()> {
        // Refresh per dictation, not on a timer: layout/group/Caps Lock can change.
        let device = u16::from(ID::USE_CORE_KBD);
        let state = self.connection.xkb_get_state(device)?.reply()?;
        let mapping = self
            .connection
            .xkb_get_map(
                device,
                MapPart::KEY_TYPES | MapPart::KEY_SYMS,
                MapPart::default(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0_u16.into(),
                0,
                0,
                0,
                0,
                0,
                0,
            )?
            .reply()?;
        let types = mapping.map.types_rtrn.ok_or("missing XKB key types")?;
        let symbols = mapping.map.syms_rtrn.ok_or("missing XKB key symbols")?;
        let modifiers = self.connection.get_modifier_mapping()?.reply()?;
        let shift = modifiers
            .keycodes
            .iter()
            .take(modifiers.keycodes_per_modifier() as usize)
            .copied()
            .find(|&code| code != 0)
            .ok_or("no Shift key in the XWayland keymap")?;
        let keys = strokes(
            mapping.first_key_sym,
            &symbols,
            &types,
            u8::from(state.group) as usize,
            u16::from(state.locked_mods) as u8,
        );
        type_text(keyboard, text, &keys, shift.into(), thread::sleep)
    }
}

fn strokes(
    first: u8,
    symbols: &[KeySymMap],
    types: &[KeyType],
    group: usize,
    locked: u8,
) -> HashMap<char, Stroke> {
    let mut keys = HashMap::new();
    // Prefer an unshifted key if more than one key can produce a character.
    for shift in [false, true] {
        for (offset, key) in symbols.iter().enumerate() {
            let groups = (key.group_info & 0x0f) as usize;
            if groups == 0 || key.width == 0 {
                continue;
            }
            let group = if group < groups {
                group
            } else {
                match key.group_info & 0xc0 {
                    0x40 => groups - 1,                              // clamp
                    0x80 => ((key.group_info & 0x30) >> 4) as usize, // redirect
                    _ => group % groups,                             // wrap
                }
            };
            let Some(kind) = key.kt_index.get(group).and_then(|&i| types.get(i as usize)) else {
                continue;
            };
            let mods = (locked | u8::from(shift)) & u16::from(kind.mods_mask) as u8;
            let level = kind
                .map
                .iter()
                .find(|entry| entry.active && u16::from(entry.mods_mask) as u8 == mods)
                .map_or(0, |entry| entry.level as usize);
            let Some(&sym) = key.syms.get(group * key.width as usize + level) else {
                continue;
            };
            if let Some(ch) = xkeysym::Keysym::new(sym)
                .key_char()
                .filter(|c| !c.is_control())
            {
                keys.entry(ch).or_insert(Stroke {
                    code: first as u16 + offset as u16,
                    shift,
                });
            }
        }
    }
    keys
}

fn type_text(
    keyboard: &mut impl Keyboard,
    text: &str,
    keys: &HashMap<char, Stroke>,
    shift: u16,
    mut wait: impl FnMut(Duration),
) -> Result<()> {
    for ch in text.chars() {
        if let Some(stroke) = keys.get(&ch) {
            if stroke.shift {
                if let Err(error) = keyboard.raw(shift, Direction::Press) {
                    // A failed roundtrip can still have delivered the press.
                    let _ = keyboard.raw(shift, Direction::Release);
                    wait(MODIFIER_SETTLE);
                    return Err(error.into());
                }
                wait(MODIFIER_SETTLE);
            }
            let typed = keyboard.raw(stroke.code, Direction::Press);
            if typed.is_ok() {
                wait(KEY_INTERVAL);
            }
            // Always release both the character and our Shift, even on failure.
            let key_released = keyboard.raw(stroke.code, Direction::Release);
            wait(KEY_INTERVAL);
            let shift_released = if stroke.shift {
                let released = keyboard.raw(shift, Direction::Release);
                wait(MODIFIER_SETTLE);
                released
            } else {
                Ok(())
            };
            typed?;
            key_released?;
            shift_released?;
        } else {
            // Preserve Enigo's Unicode fallback for symbols absent from the layout.
            // It remains compositor-dependent; don't invent an ASCII replacement.
            keyboard.text(&ch.to_string())?;
            wait(KEY_INTERVAL);
        }
    }
    Ok(())
}

#[derive(Default)]
struct Protocols {
    keyboard: bool,
}
impl Dispatch<wl_registry::WlRegistry, ()> for Protocols {
    fn event(
        state: &mut Self,
        _: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { interface, .. } = event {
            state.keyboard |= matches!(
                interface.as_str(),
                "zwp_virtual_keyboard_manager_v1" | "zwp_input_method_manager_v2"
            );
        }
    }
}
fn native_wayland_keyboard() -> bool {
    let Ok(connection) = Connection::connect_to_env() else {
        return false;
    };
    let mut queue = connection.new_event_queue();
    let mut state = Protocols::default();
    let _registry = connection.display().get_registry(&queue.handle(), ());
    queue.roundtrip(&mut state).is_ok() && state.keyboard
}

#[cfg(test)]
#[path = "output_xwayland_tests.rs"]
mod tests;
