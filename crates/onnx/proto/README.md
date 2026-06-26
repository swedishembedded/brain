# Regenerating the vendored ONNX bindings

`crates/onnx/src/onnx.rs` is **generated** from `onnx.proto` and committed
verbatim, so brain's normal build depends only on the `prost` runtime crate — no
`protoc` and no codegen run during `cargo build`.

Only regenerate when you edit `onnx.proto`. You need `protoc` on PATH and the
`prost-build` crate. A throwaway generator project:

```toml
# Cargo.toml
[package]
name = "onnx-gen"
version = "0.0.0"
edition = "2021"
[build-dependencies]
prost-build = "0.14"
```

```rust
// build.rs
fn main() {
    let mut cfg = prost_build::Config::new();
    cfg.out_dir("gen-out");
    cfg.default_package_filename("onnx");
    cfg.compile_protos(
        &["<path>/crates/onnx/proto/onnx.proto"],
        &["<path>/crates/onnx/proto"],
    ).unwrap();
}
```

`cargo build` writes `gen-out/onnx.rs`. Copy it over `crates/onnx/src/onnx.rs`
(keep the `//!` vendoring header at the top).

Field numbers and enum values in `onnx.proto` are copied verbatim from the
official `onnx/onnx` `onnx.proto`, so the encoded wire bytes are byte-identical
to a real ONNX file. `proto3` is used (prost emits plain owned structs); the wire
format is identical to the upstream `proto2` schema, and brain never relies on
serializing a default-valued singular field.
