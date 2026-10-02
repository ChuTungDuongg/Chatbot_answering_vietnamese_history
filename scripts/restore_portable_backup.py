"""Restore release payloads without overwriting different bytes; verify SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = (
    "vn_history_v1_research_runtime_2026-09-30_release.zip",
    "vn_history_v0_legacy_runtime_2026-09-30_release.zip",
    "vn_history_peft_adapter_2026-09-30_release.zip",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def destination(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if (name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative
            or not name.parts):
        raise ValueError(f"Unsafe relative path: {relative}")
    target = root.joinpath(*name.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes destination: {relative}")
    return target


def locate(root: Path, name: str) -> Path:
    candidates = [p for p in (root / name, root / "portable_backup" / name) if p.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one {name}, found {len(candidates)}")
    return candidates[0]


def inspect_archive(path: Path) -> tuple[str, dict]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP members: {path.name}")
        for info in archive.infolist():
            destination(REPO_ROOT, info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("ZIP symlinks are not supported")
        manifests = [n for n in names if n.endswith("/bundle_manifest.json")]
        if len(manifests) != 1:
            raise ValueError("Expected one bundle manifest")
        wrapper = manifests[0].split("/")[0]
        manifest = json.loads(archive.read(manifests[0]))
        expected = {wrapper + "/payload/" + f["relative_path"]: f for f in manifest["files"]}
        if len(expected) != manifest["total_files"]:
            raise ValueError("Manifest count mismatch")
        if sum(f["size_bytes"] for f in expected.values()) != manifest["total_bytes"]:
            raise ValueError("Manifest aggregate size mismatch")
        actual = {n for n in names if "/payload/" in n and not n.endswith("/")}
        if actual != set(expected):
            raise ValueError("ZIP payload differs from manifest inventory")
        checksums = {}
        for line in archive.read(wrapper + "/SHA256SUMS.txt").decode("utf-8-sig").splitlines():
            digest, name = line.split("  ", 1)
            checksums[wrapper + "/" + name] = digest
        if checksums != {n: f["sha256"] for n, f in expected.items()}:
            raise ValueError("SHA256SUMS and manifest disagree")
        for name, record in expected.items():
            if archive.getinfo(name).file_size != record["size_bytes"]:
                raise ValueError(f"ZIP size differs: {name}")
        return wrapper, manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Inspect and preflight; write nothing")
    parser.add_argument("--verify-only", action="store_true", help="Verify restored files without extracting")
    parser.add_argument("--restore-line-endings", action="store_true", help="Restore exact JSON bytes when the only existing difference is CRLF; preserve original in portable_backup/pre_restore")
    parser.add_argument("--report", type=Path, default=Path("reports/restore/restore_verification.json"))
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    bundles = [(locate(root, name),) for name in ARCHIVES]
    bundles = [(p, *inspect_archive(p)) for (p,) in bundles]
    # Check every conflict before extracting any payload.
    newline_changes = []
    for path, wrapper, manifest in bundles:
        print(f"{path.name}: {path.stat().st_size:,} ZIP bytes; {manifest['total_files']} payload files; {manifest['total_bytes']:,} payload bytes", flush=True)
        for record in manifest["files"]:
            target = destination(root, record["relative_path"])
            if target.exists():
                if not target.is_file() or target.stat().st_size != record["size_bytes"] or sha256(target) != record["sha256"]:
                    if (args.restore_line_endings and not args.verify_only and target.is_file()
                            and target.suffix == ".json" and target.stat().st_size < 100000
                            and hashlib.sha256(target.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == record["sha256"]):
                        newline_changes.append(target)
                        print(f"  Verified CRLF-only difference: {record['relative_path']}", flush=True)
                    else:
                        raise FileExistsError(f"Refusing to replace different existing bytes: {target}")
            elif args.verify_only:
                raise FileNotFoundError(target)
        print(f"  Restore payload/ relative to {root}; conflicts: none", flush=True)
    if args.dry_run:
        return 0
    for target in newline_changes:
        preserved = destination(root, "portable_backup/pre_restore/" + target.relative_to(root).as_posix())
        preserved.parent.mkdir(parents=True, exist_ok=True)
        original = target.read_bytes()
        if preserved.exists() and preserved.read_bytes() != original:
            raise FileExistsError(f"Different pre-restore backup: {preserved}")
        if not preserved.exists():
            preserved.write_bytes(original)
        temporary = target.with_name(target.name + ".restore.partial")
        with temporary.open("xb") as stream:
            stream.write(original.replace(b"\r\n", b"\n"))
        os.replace(temporary, target)
    results = []
    for path, wrapper, manifest in bundles:
        restored = skipped = 0
        with zipfile.ZipFile(path) as archive:
            for record in manifest["files"]:
                target = destination(root, record["relative_path"])
                if target.exists():
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".restore.partial")
                digest = hashlib.sha256()
                try:
                    with archive.open(wrapper + "/payload/" + record["relative_path"]) as source, temporary.open("xb") as out:
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            digest.update(block)
                            out.write(block)
                        out.flush()
                        os.fsync(out.fileno())
                    if temporary.stat().st_size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
                        raise ValueError(f"Payload SHA-256 mismatch: {record['relative_path']}")
                    # Hard-link creation is atomic and refuses a concurrently created target.
                    os.link(temporary, target)
                    restored += 1
                    print(f"  Restored + SHA-256 PASS: {record['relative_path']}", flush=True)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            if not args.verify_only:
                metadata_root = destination(root, "portable_backup/" + wrapper)
                metadata_root.mkdir(parents=True, exist_ok=True)
                for name in ("README_RESTORE.md", "bundle_manifest.json", "SHA256SUMS.txt", "restore_bundle.py"):
                    data = archive.read(wrapper + "/" + name)
                    target = metadata_root / name
                    if target.exists() and target.read_bytes() != data:
                        raise FileExistsError(f"Backup metadata differs: {target}")
                    if not target.exists():
                        target.write_bytes(data)
        result = {"archive": path.name, "archive_bytes": path.stat().st_size,
                  "archive_files": len([i for i in archive.infolist() if not i.is_dir()]),
                  "purpose": manifest["archive_kind"], "files": manifest["total_files"],
                  "bytes": manifest["total_bytes"], "restored": restored, "identical": skipped,
                  "sha256_status": "PASS", "manifest": manifest}
        results.append(result)
        print(f"PASS {path.name}: {restored} restored, {skipped} identical", flush=True)
    report = args.report if args.report.is_absolute() else destination(root, args.report.as_posix())
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
