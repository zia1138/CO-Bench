from huggingface_hub import HfFileSystem
from pathlib import Path


if __name__ == "__main__":
    fs = HfFileSystem()

    src_dir = "datasets/CO-Bench/CO-Bench"
    dst_dir = Path("./CO-Bench")

    scanned = 0
    skipped_non_config = 0
    copied = 0

    for path in fs.find(src_dir):
        scanned += 1

        # keep only config.py files
        if not path.endswith("/config.py"):
            skipped_non_config += 1
            continue

        # compute relative path
        rel_path = path[len(src_dir):].lstrip("/")
        # rename config.py to orig.py in the destination
        out_path = (dst_dir / rel_path).with_name("orig.py")

        # create directories
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # copy file
        with fs.open(path, "rb") as fsrc, open(out_path, "wb") as fdst:
            fdst.write(fsrc.read())
        copied += 1
        print(f"Copied {out_path}")

    print(
        "Done."
        f" Scanned: {scanned}, skipped non-config.py: {skipped_non_config}, copied: {copied}."
    )

