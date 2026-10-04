"""Local-only gateway metadata viewer."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=str(Path(__file__).resolve().parent / "gateway_diagnostics"))
    parser.add_argument("--analysis-id")
    parser.add_argument("--last", type=int, default=10)
    args = parser.parse_args()
    if args.last < 1:
        parser.error("--last must be positive")
    rows = []
    for path in Path(args.directory).glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("schema_version") == "gateway-diag-1.0":
                if not args.analysis_id or row.get("analysis_id") == args.analysis_id:
                    rows.append(row)
        except (ValueError, OSError):
            print("Unreadable record:", path.name)
    rows.sort(key=lambda r: r["started_at"])
    print(json.dumps(rows[-args.last:], ensure_ascii=False, indent=2))
    print("Local metadata only. Logical bytes are not token counts or exact wire bytes; non-streaming cannot separate queue/generation/network time.")


if __name__ == "__main__":
    main()
