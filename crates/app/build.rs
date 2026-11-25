//! The GPUI element tree is built by deeply nested calls on the main thread,
//! which overflows Windows' default 1 MB stack in debug builds. Give the app
//! binary 16 MB (the limit only reserves address space).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/STACK:16777216");
    }
}
