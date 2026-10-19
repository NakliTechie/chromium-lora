"""Stamp a GGUF LoRA adapter with the base model version it was built for.

The fork's backend (fork/backend/chromeml_llama.cc) computes the loaded base
model's version as "sha256:" plus the first 16 hex digits of the model file's
SHA-256, and refuses any adapter whose `chromium_lora.target_model_version`
metadata differs. This tool writes that key, using llama.cpp's own GGUF copy
routine (gguf-py, from the pinned llama.cpp checkout).

Usage:
    python3 tag_adapter.py <llama.cpp checkout> <adapter.gguf> <base model.gguf> <out.gguf>
    python3 tag_adapter.py --version-of <base model.gguf>
"""
import hashlib
import sys
from pathlib import Path

KEY = "chromium_lora.target_model_version"


def model_version(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()[:16]


def main() -> None:
    if sys.argv[1:2] == ["--version-of"]:
        print(model_version(Path(sys.argv[2])))
        return
    llama, adapter, base, out = (Path(a) for a in sys.argv[1:5])
    sys.path.insert(0, str(llama / "gguf-py"))
    import gguf
    from gguf.scripts.gguf_new_metadata import (
        MetadataDetails, copy_with_new_metadata, get_field_data)

    version = model_version(base)
    reader = gguf.GGUFReader(adapter, "r")
    arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
    writer = gguf.GGUFWriter(out, arch=arch, endianess=reader.endianess)
    alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment
    copy_with_new_metadata(
        reader, writer,
        {KEY: MetadataDetails(gguf.GGUFValueType.STRING, version)}, [])
    print(f"{out}: {KEY} = {version}")


if __name__ == "__main__":
    main()
