//! Event-driven worker wakeups. No polling timer, allocation, or application mutex in CPAL.
use std::{
    sync::OnceLock,
    thread::{self, Thread},
    time::Duration,
};

#[derive(Default)]
pub struct Wake {
    thread: OnceLock<Thread>,
}

impl Wake {
    pub fn register(&self) {
        assert!(
            self.thread.set(thread::current()).is_ok(),
            "wake registered twice"
        );
    }

    pub fn notify(&self) {
        if let Some(thread) = self.thread.get() {
            thread.unpark();
        }
    }

    pub fn wait(&self, timeout: Option<Duration>) {
        if let Some(timeout) = timeout {
            thread::park_timeout(timeout);
        } else {
            thread::park();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn notifications_before_wait_are_not_lost() {
        let wake = Wake::default();
        wake.register();
        wake.notify();
        let start = std::time::Instant::now();
        wake.wait(Some(Duration::from_secs(2)));
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}
