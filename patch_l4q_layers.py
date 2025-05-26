#!/usr/bin/env python
"""
Ganti SEMUA nn.Conv2d / torch.nn.Conv2d dan nn.Linear / torch.nn.Linear
menjadi make_conv2d / make_linear, kecuali di dalam definisi helper itu sendiri.
"""

import re, pathlib, sys

FILE = pathlib.Path("ldm/modules/diffusionmodules/model.py")

src = FILE.read_text().splitlines()
out = []

inside_make_linear = inside_make_conv2d = False
pat_linear = re.compile(r'(\W)(?:torch\.)?nn\.Linear\(')
pat_conv   = re.compile(r'(\W)(?:torch\.)?nn\.Conv2d\(')

for line in src:
    stripped = line.lstrip()

    # deteksi kita sedang berada di definisi helper
    if stripped.startswith("def make_linear"):
        inside_make_linear = True
    elif stripped.startswith("def make_conv2d"):
        inside_make_conv2d = True
    elif stripped.startswith("def "):           # fungsi lain => helper selesai
        inside_make_linear = inside_make_conv2d = False

    if not inside_make_linear:
        line = pat_linear.sub(r'\1make_linear(', line)

    if not inside_make_conv2d:
        line = pat_conv.sub(r'\1make_conv2d(', line)

    out.append(line)

dst = FILE.with_suffix(".py.l4q")     # tulis hasil patch
dst.write_text("\n".join(out))
print(f"✔ Patched file ditulis: {dst}\n"
      f"Silakan periksa sebentar; lalu timpa file asli:\n"
      f"    mv {dst} {FILE}")
