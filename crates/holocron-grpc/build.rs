fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile the proto into a tonic server (no client needed on the Rust side).
    // tonic 0.14 moved the prost generator into `tonic-prost-build`.
    tonic_prost_build::configure()
        .build_client(false)
        .build_server(true)
        .compile_protos(&["proto/holocron.proto"], &["proto"])?;
    Ok(())
}
