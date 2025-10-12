# merge_bags.py
from pathlib import Path
from time import sleep
from searchless_chess.src import bagz

def merge_bags(in_dir: str,
               pattern: str = "action_value-*-of-02148_data.bag",
               out_file: str = "action_value_data.bag") -> None:
    in_path = Path(in_dir)
    shards = sorted(in_path.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No shards matching {pattern} in {in_dir}")

    print(f"Found {len(shards)} shards. Writing -> {out_file}")

    # Prefer context manager if BagWriter supports it; otherwise wrap with try/finally.
    writer = bagz.BagWriter(out_file)
    total = 0
    try:
        for shard in shards:
            print(f"Reading {shard.name}")
            # EXPLICITLY close each reader to release file handle on Windows
            reader = bagz.BagReader(str(shard))
            try:
                for rec in reader:
                    writer.write(rec)
                    total += 1
            finally:
                # Ensure underlying file handle is released ASAP
                if hasattr(reader, "close"):
                    reader.close()
    finally:
        # Close writer; on Windows this may briefly conflict with AV / indexer
        try:
            writer.close()
        except PermissionError:
            # Retry deleting any lingering "limits.*" file with backoff
            out_path = Path(out_file)
            limits_prefix = f"limits.{out_path.name}"
            for attempt in range(10):
                # Try to remove any file named exactly limits.<out_file>
                limits_path = out_path.with_name(limits_prefix)
                removed = False
                if limits_path.exists():
                    try:
                        limits_path.unlink()
                        removed = True
                    except PermissionError:
                        sleep(0.5)
                # Some bagz versions may place limits alongside out_file; clean any matches
                for p in out_path.parent.glob("limits.*"):
                    if p.name.startswith(limits_prefix):
                        try:
                            p.unlink()
                            removed = True
                        except PermissionError:
                            pass
                if removed:
                    break
            # If still present after retries, print a hint and continue
            if limits_path.exists():
                print(f"NOTICE: Could not remove {limits_path}. "
                      f"You can delete it manually after the OS releases the lock.")

    print(f"Done. Wrote {total} records to {out_file}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing shard .bag files")
    ap.add_argument("--pattern", default="action_value-*-of-02148_data.bag")
    ap.add_argument("--out", default="action_value_data.bag")
    args = ap.parse_args()
    merge_bags(args.in_dir, args.pattern, args.out)
