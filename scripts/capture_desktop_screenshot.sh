#!/usr/bin/env bash
# Capture llm_desktop at native Retina resolution (typically ~2560px wide).
# Requires: llm_desktop running, Screen Recording permission for Terminal/Cursor.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build docs/screenshots
clang -framework Cocoa -framework CoreGraphics \
    scripts/capture_desktop_screenshot.m -o build/capture_screenshot
./build/capture_screenshot "${1:-docs/screenshots/desktop-chat.png}"
sips -g pixelWidth -g pixelHeight "${1:-docs/screenshots/desktop-chat.png}"
