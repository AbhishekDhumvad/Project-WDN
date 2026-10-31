#!/usr/bin/env python3
"""Build dist/main.exe - ONE self-contained Linux file (shell + Python) - from
    tools/main_exe_header.sh                 the shell part (prepares Python + installs the packages, then runs the rest)
    src/leak_localisation_pressure_sensors.py  the study code
    networks/*.inp, networks/*.inp.xz        the network files, packed INSIDE main.exe

Usage (from the repository root):
    python3 tools/build_main_exe.py            -> dist/main.exe
Then on the Linux workstation:
    chmod +x main.exe
    ./main.exe
"""
import base64
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NETWORKS = ["L-TOWN_v2_Real.inp", "Modena.inp", "L-TOWN.inp", "Hanoi.inp", "D-Town.inp", "L-TOWN_v2_Model.inp"]


def packed(name):
    """One EMBEDDED entry: small files zlib + base64 (list of 100-character strings), the large L-TOWN_v2_Real
    file as its .xz (lzma) data in one base64 block."""
    net = ROOT / "networks"
    if (net / (name + ".xz")).exists():
        z = base64.b64encode((net / (name + ".xz")).read_bytes()).decode()
        block = "\n".join(z[i:i + 120] for i in range(0, len(z), 120))
        return f'    "{name}": ("lzma", """\n{block}\n"""),\n'
    z = base64.b64encode(zlib.compress((net / name).read_bytes(), 9)).decode()
    rows = "".join(f'        "{z[i:i + 100]}",\n' for i in range(0, len(z), 100))
    return f'    "{name}": [\n{rows}    ],\n'


def main():
    header = (ROOT / "tools" / "main_exe_header.sh").read_text()
    src = (ROOT / "src" / "leak_localisation_pressure_sensors.py").read_text()
    src = src.split("\n", 1)[1] if src.startswith("#!") else src          # the shell header has its own first line
    marker = "EMBEDDED = {}\n"
    assert src.count(marker) == 1, "EMBEDDED placeholder not found in the source"
    embedded = ("# EMBEDDED NETWORK FILES (zlib + base64; L-TOWN_v2_Real: the original 177 MB file, lzma) - written next to\n"
                "# main.exe when needed.\nEMBEDDED = {\n" + "".join(packed(n) for n in NETWORKS) + "}\n")
    out = ROOT / "dist" / "main.exe"
    out.parent.mkdir(exist_ok=True)
    out.write_text(header + "\n" + src.replace(marker, embedded))
    out.chmod(0o755)
    print(f"written {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
