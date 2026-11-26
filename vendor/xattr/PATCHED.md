# Patched copy of xattr 0.2.3

This is `xattr` 0.2.3 from crates.io (MIT OR Apache-2.0, see the licence files here) with one
change in `src/sys/mod.rs`: the "no such attribute" constant is `ENODATA` on Linux and Android
instead of `libc::ENOATTR`, which current `libc` releases no longer define there.

It is a transitive dependency (GPUI -> zed-async-tar), and `zed-async-tar` needs `xattr ^0.2`, so a
newer release cannot be used. Remove this directory and the `[patch.crates-io]` entry in the
root `Cargo.toml` once GPUI moves to a fixed `xattr`.
