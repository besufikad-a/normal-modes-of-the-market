"""Concatenate rendered scenes into one film (H.264, no ffmpeg binary needed).

Usage: python stitch.py <quality_dir> <out.mp4>   e.g.  1080p30 normal_modes_of_the_market.mp4
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import av

ORDER = [
    ("s0_title", "S0_Title"),
    ("s1_history", "S1_History"),
    ("s1_history", "S2_Relationship3D"),
    ("s1_history", "S3_CorrelationRibbons"),
    ("s2_oscillators", "S4_TwoOscillators"),
    ("s2_oscillators", "S5_ChainModes"),
    ("s2_oscillators", "S6_ModeSurface3D"),
    ("s3_market", "S7_MarketModes"),
    ("s3_market", "S8_Spectrum3D"),
    ("s3_market", "S9_Findings"),
]


def main(quality: str, out: str) -> None:
    root = Path(__file__).resolve().parent / "media" / "videos"
    files = [root / mod / quality / f"{scene}.mp4" for mod, scene in ORDER]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    first = av.open(str(files[0])).streams.video[0]
    w, h = first.codec_context.width, first.codec_context.height
    fps = int(round(float(first.average_rate)))
    dst = av.open(out, "w")
    st = dst.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": "23", "preset": "medium"}
    tb = Fraction(1, fps)
    n = 0
    for f in files:
        src = av.open(str(f))
        for frame in src.decode(video=0):
            frame = frame.reformat(width=w, height=h, format="yuv420p")
            frame.pts = n
            frame.time_base = tb
            n += 1
            for pkt in st.encode(frame):
                dst.mux(pkt)
        src.close()
        print(f"  + {f.name}: total {n / fps:.1f}s")
    for pkt in st.encode():
        dst.mux(pkt)
    dst.close()
    print(f"wrote {out} ({n / fps:.1f}s, {w}x{h}@{fps})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
