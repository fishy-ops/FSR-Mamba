"""Embed validated DXIL without a runtime compiler dependency."""
import argparse
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("input", type=Path)
p.add_argument("output", type=Path)
p.add_argument("symbol")
p.add_argument("--compiler", type=Path)
p.add_argument("--source", type=Path)
p.add_argument("--include", type=Path)
p.add_argument("--define", action="append", default=[])
a = p.parse_args()
if a.compiler:
    if not a.source or not a.include:
        p.error("--compiler requires --source and --include")
    if a.compiler.stem == "dxc_host":
        # The minimal host has no -D parser; compile an equivalent preprocessor wrapper.
        source = a.source
        if a.define:
            wrapper = a.input.with_suffix(".hlsl")
            wrapper.write_text("".join("#define " + d.replace("=", " ", 1) + "\n" for d in a.define) +
                               '#include "' + a.source.resolve().as_posix() + '"\n')
            source = wrapper
        subprocess.run([str(a.compiler), str(source), str(a.include), str(a.input)], check=True)
    else:
        command = [str(a.compiler), "-T", "cs_6_0", "-E", "main", "-I", str(a.include), "-Gis", "-WX", "-O3"]
        for define in a.define:
            command += ["-D", define]
        subprocess.run(command + ["-Fo", str(a.input), str(a.source)], check=True)
blob = a.input.read_bytes()
if not blob.startswith(b"DXBC"):
    raise ValueError("not a DXIL container")
a.output.write_text("#pragma once\nstatic const unsigned char " + a.symbol + "[] = {\n" +
                    "\n".join(",".join(str(v) for v in blob[i:i+32]) + "," for i in range(0, len(blob), 32)) + "\n};\n")
