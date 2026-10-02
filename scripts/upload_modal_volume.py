"""Upload extracted runtime files using Modal 1.5 SDK, refusing remote conflicts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

from scripts.modal_artifact_checks import app, audit_volume
from scripts.restore_portable_backup import ARCHIVES, REPO_ROOT, inspect_archive, locate, sha256


def inventory(include_v0: bool = False) -> list[dict]:
    records = []
    for name in ARCHIVES:
        archive = locate(REPO_ROOT, name)
        wrapper, manifest = inspect_archive(archive)
        for item in manifest["files"]:
            relative = item["relative_path"]
            if not (relative.startswith("artifacts/corpus_v1/") or relative.startswith("artifacts/models/")
                    or (include_v0 and relative.startswith("artifacts/vn_history_deployment/"))):
                continue
            remote = "/" + relative.removeprefix("artifacts/")
            if relative.startswith("artifacts/vn_history_deployment/"):
                # Preserve the pre-existing older bundle; release V0 is a separate snapshot.
                remote = "/v0/" + relative.removeprefix("artifacts/vn_history_deployment/")
            records.append({**item, "remote_path": remote})
        for metadata in ("bundle_manifest.json", "SHA256SUMS.txt"):
            path = REPO_ROOT / "portable_backup" / wrapper / metadata
            records.append({"relative_path": path.relative_to(REPO_ROOT).as_posix(),
                            "remote_path": f"/restore_manifests/{wrapper}/{metadata}",
                            "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return records


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-v0", action="store_true", help="Keep release V0 in /v0, preserving any older remote bundle")
    parser.add_argument("--dry-run", action="store_true", help="Check local files and list upload mapping; no cloud changes")
    parser.add_argument("--verify-only", action="store_true", help="Hash all remote files without uploading")
    parser.add_argument("--ipv4", action="store_true", help="Resolve upload/control hostnames over IPv4 in this process only")
    args = parser.parse_args(argv)
    if args.ipv4:
        import socket
        original_getaddrinfo = socket.getaddrinfo

        def ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            if family == socket.AF_UNSPEC:
                family = socket.AF_INET
            return original_getaddrinfo(host, port, family, type, proto, flags)

        socket.getaddrinfo = ipv4_getaddrinfo
        print("Network: process-local IPv4 hostname resolution", flush=True)
    records = inventory(args.include_v0)
    for item in records:
        local = REPO_ROOT / item["relative_path"]
        if not local.is_file() or local.stat().st_size != item["size_bytes"] or sha256(local) != item["sha256"]:
            raise RuntimeError(f"Local manifest mismatch: {item['relative_path']}")
        print(f"{item['relative_path']} -> vn-history-artifacts:{item['remote_path']} ({item['size_bytes']:,} bytes)", flush=True)
    total = sum(item["size_bytes"] for item in records)
    print(f"Total: {len(records)} files / {total:,} bytes", flush=True)
    if args.dry_run:
        return 0
    with modal.enable_output(), app.run():
        before = audit_volume.remote(records)
    conflicts = [item for item in before if item["status"] == "DIFFERENT"]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite remote files: {conflicts}")
    missing = {item["remote_path"] for item in before if item["status"] == "MISSING"}
    if args.verify_only and missing:
        raise FileNotFoundError(f"Missing remote paths: {sorted(missing)}")
    if missing:
        volume = modal.Volume.from_name("vn-history-artifacts", create_if_missing=False)
        # Commit each file so an interrupted transfer can resume without re-upload.
        # Avoid keeping an App heartbeat open during large transfers on this network.
        for item in sorted(records, key=lambda item: item["size_bytes"]):
            if item["remote_path"] in missing:
                print(f"Uploading {item['remote_path']} ({item['size_bytes']:,} bytes)", flush=True)
                with modal.enable_output(), volume.batch_upload(force=False) as batch:
                    batch.put_file(REPO_ROOT / item["relative_path"], item["remote_path"])
                print(f"Committed {item['remote_path']}", flush=True)
    if missing:
        with modal.enable_output(), app.run():
            after = audit_volume.remote(records)
    else:
        after = before
    if any(item["status"] != "PASS" for item in after):
        raise RuntimeError(f"Remote SHA-256 verification failed: {after}")
    report = {"volume": "vn-history-artifacts", "mount": "/artifacts", "files": len(records),
              "bytes": total, "new_files": len(missing), "uploaded_bytes": sum(item["size_bytes"] for item in records if item["remote_path"] in missing),
              "verification": after, "records": records}
    target = REPO_ROOT / "reports/restore/modal_volume_verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(records)} remote files; {total:,} bytes; all SHA-256 match", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
