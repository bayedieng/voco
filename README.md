# voco

Rust workspace containing:

- `voco-lib/`: shared library crate (imported as `voco_lib`).
- `vocod/`: binary crate, depending on `voco-lib`.
- `xtask/`: development automation using the [xtask convention](https://github.com/matklad/cargo-xtask).

## Build

```sh
cargo xtask build
cargo xtask build --release
```

The `cargo xtask` alias runs the local `xtask` crate. Its build task builds
`voco-lib` and `vocod`, forwarding additional arguments to `cargo build`.
No globally installed task runner is needed.

## Development

```sh
cargo run --package vocod
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```
