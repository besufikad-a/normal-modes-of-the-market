#!/usr/bin/env bash
# Render every scene at 1080p30 and stitch them into one film.
# Usage (from animations/):  bash render_all.sh [manim-binary]
set -euo pipefail
cd "$(dirname "$0")"
MANIM=${1:-manim}
Q="--resolution 1920,1080 --frame_rate 30 --disable_caching"
render() { echo "== $2"; $MANIM $Q "$1" "$2" > "logs_$2.txt" 2>&1; }

render s0_title.py S0_Title &
render s1_history.py S1_History &
render s2_oscillators.py S4_TwoOscillators &
render s2_oscillators.py S5_ChainModes &
wait
render s1_history.py S2_Relationship3D &
render s1_history.py S3_CorrelationRibbons &
render s3_market.py S7_MarketModes &
render s3_market.py S9_Findings &
wait
render s2_oscillators.py S6_ModeSurface3D &
render s3_market.py S8_Spectrum3D &
wait

PY=$(dirname "$MANIM")/python
[ -x "$PY" ] || PY=python
"$PY" stitch.py 1080p30 normal_modes_of_the_market.mp4
