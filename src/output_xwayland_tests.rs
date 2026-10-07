use super::*;
use enigo::{InputError, InputResult, Key};
use std::{cell::Cell, rc::Rc};
use x11rb::protocol::{xkb::KTMapEntry, xproto::ModMask};

fn key(base: char, upper: char) -> KeySymMap {
    KeySymMap {
        kt_index: [0; 4],
        group_info: 1,
        width: 2,
        syms: vec![base as u32, upper as u32],
    }
}
fn two_level() -> KeyType {
    KeyType {
        mods_mask: ModMask::SHIFT,
        num_levels: 2,
        map: vec![KTMapEntry {
            active: true,
            mods_mask: ModMask::SHIFT,
            level: 1,
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[derive(Debug, PartialEq)]
enum Event {
    Raw(u16, Direction),
    Text(String),
}
#[derive(Default)]
struct FakeKeyboard {
    events: Vec<Event>,
    fail_code: Option<u16>,
    event_count: Rc<Cell<usize>>,
}
impl Keyboard for FakeKeyboard {
    fn fast_text(&mut self, text: &str) -> InputResult<Option<()>> {
        self.events.push(Event::Text(text.to_owned()));
        self.event_count.set(self.events.len());
        Ok(Some(()))
    }
    fn key(&mut self, _: Key, _: Direction) -> InputResult<()> {
        panic!("must not use Enigo's dynamically remapped Unicode keys");
    }
    fn raw(&mut self, code: u16, direction: Direction) -> InputResult<()> {
        self.events.push(Event::Raw(code, direction));
        self.event_count.set(self.events.len());
        if self.fail_code == Some(code) {
            Err(InputError::Simulate("test failure"))
        } else {
            Ok(())
        }
    }
}

#[test]
fn capitals_and_shifted_punctuation_use_existing_keys() -> Result<()> {
    let keys = strokes(
        20,
        &[key('t', 'T'), key('h', 'H'), key('/', '?')],
        &[two_level()],
        0,
        0,
    );
    let mut keyboard = FakeKeyboard::default();
    type_text(&mut keyboard, "Th?T", &keys, 50, |_| {})?;
    use Direction::*;
    assert_eq!(
        keyboard.events,
        [
            Event::Raw(50, Press),
            Event::Raw(20, Press),
            Event::Raw(20, Release),
            Event::Raw(50, Release),
            Event::Raw(21, Press),
            Event::Raw(21, Release),
            Event::Raw(50, Press),
            Event::Raw(22, Press),
            Event::Raw(22, Release),
            Event::Raw(50, Release),
            Event::Raw(50, Press),
            Event::Raw(20, Press),
            Event::Raw(20, Release),
            Event::Raw(50, Release),
        ]
    );
    Ok(())
}

#[test]
fn shift_is_released_even_if_typing_fails() {
    let keys = strokes(20, &[key('t', 'T')], &[two_level()], 0, 0);
    let mut keyboard = FakeKeyboard {
        fail_code: Some(20),
        ..Default::default()
    };
    assert!(type_text(&mut keyboard, "T", &keys, 50, |_| {}).is_err());
    assert_eq!(
        keyboard.events.last(),
        Some(&Event::Raw(50, Direction::Release))
    );
}

#[test]
fn unicode_fallback_and_repeated_letters_are_preserved() -> Result<()> {
    let keys = strokes(20, &[key('t', 'T')], &[two_level()], 0, 0);
    let mut keyboard = FakeKeyboard::default();
    type_text(&mut keyboard, "tté", &keys, 50, |_| {})?;
    assert_eq!(
        keyboard.events,
        [
            Event::Raw(20, Direction::Press),
            Event::Raw(20, Direction::Release),
            Event::Raw(20, Direction::Press),
            Event::Raw(20, Direction::Release),
            Event::Text("é".into()),
        ]
    );
    Ok(())
}

#[test]
fn modifier_updates_settle_before_uppercase_and_lowercase_keys() -> Result<()> {
    let keys = strokes(20, &[key('t', 'T'), key('h', 'H')], &[two_level()], 0, 0);
    let mut keyboard = FakeKeyboard::default();
    let count = Rc::clone(&keyboard.event_count);
    let mut pauses = Vec::new();
    type_text(&mut keyboard, "Th", &keys, 50, |duration| {
        pauses.push((count.get(), duration));
    })?;
    assert_eq!(
        pauses,
        [
            (1, MODIFIER_SETTLE), // Shift down -> settle -> T down
            (2, KEY_INTERVAL),    // T down -> hold -> T up
            (3, KEY_INTERVAL),    // T up -> settle -> Shift up
            (4, MODIFIER_SETTLE), // Shift up -> settle -> h down
            (5, KEY_INTERVAL),
            (6, KEY_INTERVAL),
        ]
    );
    Ok(())
}

#[test]
fn failed_shift_press_still_attempts_release() {
    let keys = strokes(20, &[key('t', 'T')], &[two_level()], 0, 0);
    let mut keyboard = FakeKeyboard {
        fail_code: Some(50),
        ..Default::default()
    };
    let mut pauses = Vec::new();
    assert!(type_text(&mut keyboard, "T", &keys, 50, |d| pauses.push(d)).is_err());
    assert_eq!(
        keyboard.events,
        [
            Event::Raw(50, Direction::Press),
            Event::Raw(50, Direction::Release),
        ]
    );
    assert_eq!(pauses, [MODIFIER_SETTLE]);
}

#[test]
fn active_layout_group_and_non_us_punctuation_are_respected() {
    let mut french = key('a', 'A');
    french.group_info = 2;
    french.syms.extend(['q' as u32, 'Q' as u32]);
    let keys = strokes(20, &[french, key(',', '?')], &[two_level()], 1, 0);
    assert_eq!(
        keys[&'Q'],
        Stroke {
            code: 20,
            shift: true
        }
    );
    assert!(!keys.contains_key(&'A'));
    assert_eq!(
        keys[&'?'],
        Stroke {
            code: 21,
            shift: true
        }
    );
}

#[test]
fn caps_lock_changes_which_letters_need_shift() {
    let mut alphabetic = two_level();
    alphabetic.mods_mask = ModMask::SHIFT | ModMask::LOCK;
    alphabetic.map.push(KTMapEntry {
        active: true,
        mods_mask: ModMask::LOCK,
        level: 1,
        ..Default::default()
    });
    let keys = strokes(
        20,
        &[key('t', 'T')],
        &[alphabetic],
        0,
        u16::from(ModMask::LOCK) as u8,
    );
    assert_eq!(
        keys[&'T'],
        Stroke {
            code: 20,
            shift: false
        }
    );
    assert_eq!(
        keys[&'t'],
        Stroke {
            code: 20,
            shift: true
        }
    );
}

#[test]
#[ignore = "read-only inspection of a live XWayland keymap; needs DISPLAY and WAYLAND_DISPLAY"]
fn desktop_keymap_can_plan_reported_missing_letters() -> Result<()> {
    let keys = XwaylandKeys::detect()?.ok_or("not an XWayland fallback session")?;
    let mut keyboard = FakeKeyboard::default();
    keys.text(&mut keyboard, "It's. This. Do you like it? TTT")?;
    assert!(
        keyboard
            .events
            .iter()
            .all(|event| matches!(event, Event::Raw(..))),
        "{:?}",
        keyboard.events
    );
    Ok(())
}
