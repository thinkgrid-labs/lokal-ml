#!/usr/bin/env bash
# build-ios.sh — Compile lokal-ml-ffi into an XCFramework for iOS.
#
# Outputs:
#   ios/LokalML.xcframework   — fat XCFramework (device + simulator slices)
#   cpp/lokal-ml.h            — regenerated C header via cbindgen
#
# Prerequisites: rustup, cbindgen (`cargo install cbindgen`), Xcode CLT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CRATE_DIR="$PACKAGE_DIR/rust"
HEADER_DIR="$PACKAGE_DIR/cpp"
IOS_DIR="$PACKAGE_DIR/ios"

# Workspace target directory: three levels up from the crate is the workspace root.
WORKSPACE_ROOT="$(cd "$CRATE_DIR/../../.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$WORKSPACE_ROOT/target}"

TARGETS_DEVICE="aarch64-apple-ios"
TARGETS_SIM_ARM="aarch64-apple-ios-sim"
TARGETS_SIM_X86="x86_64-apple-ios"

echo "==> Adding Rust targets"
for t in $TARGETS_DEVICE $TARGETS_SIM_ARM $TARGETS_SIM_X86; do
  rustup target add "$t" 2>/dev/null || true
done

echo "==> Building device slice ($TARGETS_DEVICE)"
cargo build --release \
  --target "$TARGETS_DEVICE" \
  --manifest-path "$CRATE_DIR/Cargo.toml"

echo "==> Building simulator slices ($TARGETS_SIM_ARM + $TARGETS_SIM_X86)"
cargo build --release \
  --target "$TARGETS_SIM_ARM" \
  --manifest-path "$CRATE_DIR/Cargo.toml"
cargo build --release \
  --target "$TARGETS_SIM_X86" \
  --manifest-path "$CRATE_DIR/Cargo.toml"

echo "==> Lipo-ing simulator slices"
mkdir -p "$IOS_DIR/simulator" "$IOS_DIR/device"
lipo -create \
  "$TARGET_DIR/$TARGETS_SIM_ARM/release/liblokal_ml_ffi.a" \
  "$TARGET_DIR/$TARGETS_SIM_X86/release/liblokal_ml_ffi.a" \
  -output "$IOS_DIR/simulator/liblokal_ml_ffi.a"

cp "$TARGET_DIR/$TARGETS_DEVICE/release/liblokal_ml_ffi.a" \
   "$IOS_DIR/device/liblokal_ml_ffi.a"

echo "==> Generating C header with cbindgen"
mkdir -p "$HEADER_DIR"
cbindgen \
  --config "$CRATE_DIR/cbindgen.toml" \
  --crate lokal-ml-ffi \
  --output "$HEADER_DIR/lokal-ml.h" \
  "$CRATE_DIR"

echo "==> Creating XCFramework"
rm -rf "$IOS_DIR/LokalML.xcframework"
xcodebuild -create-xcframework \
  -library "$IOS_DIR/device/liblokal_ml_ffi.a"    -headers "$HEADER_DIR" \
  -library "$IOS_DIR/simulator/liblokal_ml_ffi.a" -headers "$HEADER_DIR" \
  -output  "$IOS_DIR/LokalML.xcframework"

echo ""
echo "Done. XCFramework at: $IOS_DIR/LokalML.xcframework"
echo "Header at:            $HEADER_DIR/lokal-ml.h"
