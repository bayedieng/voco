//! Carbon hotkeys need Carbon event dispatch, not merely CFRunLoopRun.
use crate::{Result, control::Control};
use std::{
    ffi::c_void,
    ptr,
    sync::{OnceLock, atomic::Ordering},
};
type Event = *mut c_void;
static QUEUE: OnceLock<usize> = OnceLock::new();
const WAKE: u32 = u32::from_be_bytes(*b"voco");

#[link(name = "Carbon", kind = "framework")]
unsafe extern "C" {
    fn GetMainEventQueue() -> *mut c_void;
    fn GetEventDispatcherTarget() -> *mut c_void;
    fn ReceiveNextEvent(
        count: u32,
        types: *const c_void,
        timeout: f64,
        pull: u8,
        event: *mut Event,
    ) -> i32;
    fn SendEventToEventTarget(event: Event, target: *mut c_void) -> i32;
    fn ReleaseEvent(event: Event);
    fn CreateEvent(
        allocator: *const c_void,
        class: u32,
        kind: u32,
        time: f64,
        attributes: u32,
        event: *mut Event,
    ) -> i32;
    fn PostEventToQueue(queue: *mut c_void, event: Event, priority: i16) -> i32;
}

pub fn prepare() -> Result<()> {
    // SAFETY: called on the same main thread that created the global hotkey manager.
    let queue = unsafe { GetMainEventQueue() };
    if queue.is_null() {
        return Err("macOS application event queue unavailable".into());
    }
    let _ = QUEUE.set(queue as usize);
    Ok(())
}

pub fn run(control: &Control) -> Result<()> {
    while !control.stop.load(Ordering::Acquire) {
        let mut event = ptr::null_mut();
        // SAFETY: main-thread Carbon event pumping; count=0 accepts all events, -1 waits forever.
        let status = unsafe { ReceiveNextEvent(0, ptr::null(), -1.0, 1, &mut event) };
        if status != 0 {
            return Err(format!("macOS event receive failed: {status}").into());
        }
        if !event.is_null() {
            // SAFETY: this owned event is dispatched and released once. Unhandled wake events are OK.
            unsafe {
                SendEventToEventTarget(event, GetEventDispatcherTarget());
                ReleaseEvent(event);
            }
        }
    }
    Ok(())
}

pub fn wake() {
    let Some(&queue) = QUEUE.get() else {
        return;
    };
    let mut event = ptr::null_mut();
    // SAFETY: Carbon event creation/posting is thread-safe; the process-owned main queue remains
    // valid. Posting retains the event. A queued wake also survives shutdown just before run().
    unsafe {
        if CreateEvent(ptr::null(), WAKE, 1, 0.0, 0, &mut event) == 0 && !event.is_null() {
            PostEventToQueue(queue as *mut c_void, event, 2);
            ReleaseEvent(event);
        }
    }
}
