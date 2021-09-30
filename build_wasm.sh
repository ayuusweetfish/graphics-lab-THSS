cargo build --release -p polygon_intersect --lib --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/polygon_intersect.wasm --out-dir web --no-modules --no-typescript
