#!/usr/bin/env python3
"""
sb3_model_converter.py

Converts a stable-baselines3 model saved with NumPy 2.x to be compatible
with NumPy 1.26.x by rewriting pickle namespace references at the opcode level.

Usage:
    python3 sb3_model_converter.py model.zip model_converted.zip
"""

import zipfile
import shutil
import pickle
import struct
import io
import re
import sys
import os
from pathlib import Path


# ── Pickle opcode constants ──────────────────────────────────────────────────
# These are the raw opcode bytes used in the pickle protocol.
SHORT_BINUNICODE = b'\x8c'  # opcode for strings < 256 chars (protocol 4)
BINUNICODE       = b'X'     # opcode for longer unicode strings
GLOBAL           = b'c'     # opcode for GLOBAL (module\nname\n)
STACK_GLOBAL     = b'\x93'  # opcode for STACK_GLOBAL (protocol 4+)


def rewrite_pickle_bytes(data: bytes) -> bytes:
    """
    Scan a raw pickle bytestream and rewrite all numpy._core references
    to numpy.core, without deserializing anything.

    Handles all pickle protocols (2, 4, 5) and both GLOBAL and
    STACK_GLOBAL opcodes.
    """
    # Strategy 1: Direct byte substitution for encoded string opcodes.
    # In protocols 4+, module/attribute names are pushed as SHORT_BINUNICODE
    # or BINUNICODE strings before STACK_GLOBAL. We can safely replace the
    # raw UTF-8 bytes because the string length prefix will remain the same
    # (numpy._core and numpy.core differ by 1 char, so we handle length too).

    out = bytearray()
    stream = io.BytesIO(data)

    while True:
        opcode = stream.read(1)
        if not opcode:
            break

        # ── SHORT_BINUNICODE: \x8c <1-byte-len> <utf8> ──────────────────
        if opcode == SHORT_BINUNICODE:
            length_byte = stream.read(1)
            length = length_byte[0]
            payload = stream.read(length)
            text = payload.decode('utf-8', errors='replace')

            if 'numpy._core' in text:
                new_text = text.replace('numpy._core', 'numpy.core')
                new_payload = new_text.encode('utf-8')
                new_length = len(new_payload)
                if new_length > 255:
                    # Promote to BINUNICODE (4-byte length)
                    out += b'X' + struct.pack('<I', new_length) + new_payload
                else:
                    out += SHORT_BINUNICODE + bytes([new_length]) + new_payload
            else:
                out += opcode + length_byte + payload

        # ── BINUNICODE: X <4-byte-le-len> <utf8> ────────────────────────
        elif opcode == BINUNICODE:
            length_bytes = stream.read(4)
            length = struct.unpack('<I', length_bytes)[0]
            payload = stream.read(length)
            text = payload.decode('utf-8', errors='replace')

            if 'numpy._core' in text:
                new_text = text.replace('numpy._core', 'numpy.core')
                new_payload = new_text.encode('utf-8')
                out += b'X' + struct.pack('<I', len(new_payload)) + new_payload
            else:
                out += opcode + length_bytes + payload

        # ── GLOBAL opcode: c<module>\n<name>\n ──────────────────────────
        # Used in protocols 0-2. Reads two newline-terminated strings.
        elif opcode == GLOBAL:
            module_line = b''
            while True:
                ch = stream.read(1)
                if ch == b'\n' or not ch:
                    break
                module_line += ch
            name_line = b''
            while True:
                ch = stream.read(1)
                if ch == b'\n' or not ch:
                    break
                name_line += ch

            module_str = module_line.decode('utf-8', errors='replace')
            name_str   = name_line.decode('utf-8', errors='replace')

            module_str = module_str.replace('numpy._core', 'numpy.core')
            name_str   = name_str.replace('numpy._core', 'numpy.core')

            out += GLOBAL
            out += module_str.encode('utf-8') + b'\n'
            out += name_str.encode('utf-8') + b'\n'

        else:
            out += opcode

    return bytes(out)


def convert_sb3_zip(input_path: str, output_path: str) -> None:
    """
    Open an SB3 .zip model, rewrite all pickle files inside it,
    and save a new .zip that is NumPy 1.x compatible.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Model not found: {input_path}")

    print(f"[+] Opening model: {input_path}")

    with zipfile.ZipFile(input_path, 'r') as zin, \
         zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:

        for item in zin.infolist():
            raw = zin.read(item.name)

            # Only rewrite files that are pickle streams.
            # SB3 stores the main metadata in a file called "data"
            # (no extension) which is a raw pickle. policy.pth is a
            # PyTorch zip-of-pickles and is handled separately below.
            if item.name == 'data':
                print(f"    [~] Rewriting pickle namespace in: {item.name}")
                raw = rewrite_pickle_bytes(raw)

            elif item.name.endswith('.pkl'):
                print(f"    [~] Rewriting pickle namespace in: {item.name}")
                raw = rewrite_pickle_bytes(raw)

            elif item.name == 'policy.pth':
                # policy.pth is itself a zip file (PyTorch's format).
                # We need to recursively enter it and rewrite its pickles.
                print(f"    [~] Rewriting PyTorch archive: {item.name}")
                raw = rewrite_pytorch_archive(raw)

            else:
                print(f"    [=] Copying unchanged: {item.name}")

            zout.writestr(item, raw)

    print(f"\n[✓] Converted model saved to: {output_path}")


def rewrite_pytorch_archive(pth_bytes: bytes) -> bytes:
    """
    A PyTorch .pth file is itself a ZIP archive containing pickle files
    (e.g. archive/data.pkl). Recursively rewrite numpy refs inside it.
    """
    in_buf  = io.BytesIO(pth_bytes)
    out_buf = io.BytesIO()

    with zipfile.ZipFile(in_buf, 'r') as zin, \
         zipfile.ZipFile(out_buf, 'w', compression=zipfile.ZIP_DEFLATED) as zout:

        for item in zin.infolist():
            raw = zin.read(item.name)

            if item.name.endswith('.pkl') or item.name.endswith('/data.pkl'):
                raw = rewrite_pickle_bytes(raw)
                print(f"        [~] Rewrote: {item.name}")
            else:
                print(f"        [=] Copied:  {item.name}")

            zout.writestr(item, raw)

    return out_buf.getvalue()


def verify_conversion(model_path: str) -> None:
    """
    Quick sanity check: scan the output zip for any remaining
    numpy._core references in pickle files.
    """
    print(f"\n[+] Verifying: {model_path}")
    found = False

    with zipfile.ZipFile(model_path, 'r') as z:
        for item in z.infolist():
            raw = z.read(item.name)
            if b'numpy._core' in raw:
                print(f"    [!] WARNING — numpy._core still found in: {item.name}")
                found = True

    if not found:
        print("    [✓] Clean — no numpy._core references remain.")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 sb3_model_converter.py <input.zip> <output.zip>")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2]

    convert_sb3_zip(src, dst)
    verify_conversion(dst)

    print("\n[→] You can now load the converted model with:")
    print(f"      model = PPO.load('{dst}')")