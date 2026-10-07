//! Lazily reuse ASR sessions and return their freed heap pages after inactivity.
use crate::Result;
use std::time::{Duration, Instant};

pub struct ModelCache<T> {
    model: Option<T>,
    last_used: Instant,
    timeout: Option<Duration>,
}

impl<T> ModelCache<T> {
    /// None keeps the model resident; a timeout releases it between dictation sessions.
    pub fn new(timeout: Option<Duration>) -> Self {
        Self {
            model: None,
            last_used: Instant::now(),
            timeout,
        }
    }

    pub fn get_or_load(&mut self, load: impl FnOnce() -> Result<T>) -> Result<&mut T> {
        if self.model.is_none() {
            self.model = Some(load()?);
        }
        self.touch(Instant::now());
        Ok(self.model.as_mut().expect("loaded"))
    }

    pub fn touch(&mut self, now: Instant) {
        self.last_used = now;
    }

    pub fn expire(&mut self, now: Instant, speech_active: bool) -> bool {
        if !speech_active
            && self.model.is_some()
            && self
                .timeout
                .is_some_and(|timeout| now.duration_since(self.last_used) >= timeout)
        {
            self.model = None;
            reclaim_unused_memory();
            return true;
        }
        false
    }

    pub fn wait_timeout(&self, now: Instant, speech_active: bool) -> Option<Duration> {
        if speech_active || self.model.is_none() {
            return None;
        }
        self.timeout
            .map(|timeout| timeout.saturating_sub(now.duration_since(self.last_used)))
    }
}

/// Session drop frees weights, but system allocators can retain hundreds of MB of free pages.
/// Run once on eviction, never on audio callbacks or during inference.
fn reclaim_unused_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> std::ffi::c_int;
        }
        // SAFETY: glibc's thread-safe API trims free pages without invalidating live allocations.
        malloc_trim(0);
    }
    #[cfg(target_os = "macos")]
    unsafe {
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        // SAFETY: a null zone selects all malloc zones; goal=0 reclaims available free pages.
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lazy_reused_expired_and_reloaded() -> Result<()> {
        let mut cache = ModelCache::new(Some(Duration::from_secs(10)));
        assert!(cache.wait_timeout(Instant::now(), false).is_none());
        cache.get_or_load(|| Ok(42))?;
        assert_eq!(*cache.get_or_load(|| panic!("must reuse"))?, 42);
        let now = Instant::now();
        cache.touch(now);
        assert!(!cache.expire(now + Duration::from_secs(9), false));
        assert!(!cache.expire(now + Duration::from_secs(11), true));
        assert!(
            cache
                .wait_timeout(now + Duration::from_secs(11), true)
                .is_none()
        );
        assert!(cache.expire(now + Duration::from_secs(11), false));
        assert_eq!(*cache.get_or_load(|| Ok(7))?, 7);
        Ok(())
    }
    #[test]
    fn resident_mode_never_expires() -> Result<()> {
        let mut cache = ModelCache::new(None);
        cache.get_or_load(|| Ok(1))?;
        assert!(!cache.expire(Instant::now() + Duration::from_secs(3600), false));
        assert!(cache.wait_timeout(Instant::now(), false).is_none());
        Ok(())
    }
}
