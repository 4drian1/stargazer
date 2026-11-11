"""Reversible asset migration (#18.199): copy -> verify -> marker -> switch ->
validate. The old copy is never deleted here; disposition happens separately
under explicit authorization after the new location proves itself.
"""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate(src, dst, switch, validate):
    """Migrate one asset folder.

    src: existing asset folder (complete marker required).
    dst: new versioned folder (must not exist).
    switch: callable(dst) reconfiguring readers to the new location; runs only
            after the copied asset validates with its complete marker.
    validate: callable(dst) -> truthy, evaluated BEFORE the switch.
    Returns a receipt dict. Raises without switching when any step fails."""
    src, dst = Path(src), Path(dst)
    if not (src / 'COMPLETE.json').is_file():
        raise ValueError('SOURCE_NOT_COMPLETE:' + src.name)
    if dst.exists():
        raise ValueError('TARGET_EXISTS:' + dst.name)
    lock = dst.with_name(dst.name + '.writing.lock')
    if lock.exists():
        raise ValueError('ASSET_BUSY:' + dst.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = dst.with_name(dst.name + '.migrating')
    try:
        lock.open('xb').close()
    except FileExistsError:
        raise ValueError('ASSET_BUSY:' + dst.name)
    # The staging cleanup happens only AFTER our exclusive lock is acquired:
    # a concurrent migrator's staging is never touched.
    if staging.exists():
        shutil.rmtree(staging)
    try:
        shutil.copytree(src, staging)
        # Hash every file EXCEPT the marker; the rewritten marker then stays
        # self-consistent (its own digest is not part of the manifest).
        manifest = []
        for f in sorted(staging.rglob('*')):
            if f.is_file() and f.name != 'COMPLETE.json':
                manifest.append([str(f.relative_to(staging)).replace('\\', '/'),
                                 _sha(f)])
        old_marker = json.loads((staging / 'COMPLETE.json').read_bytes())
        old_files = old_marker.get('files')
        if old_files is not None:
            # Both marker shapes are accepted: [[name, sha], ...] and the
            # prefix-cache dict {name: sha}. Coverage must match exactly —
            # the staged copy may hold nothing beyond the listed files.
            expected = (dict(old_files) if isinstance(old_files, dict)
                        else {name: sha for name, sha in old_files})
            actual = {name: sha for name, sha in manifest}
            staged_extra = {str(f.relative_to(staging)).replace('\\', '/')
                            for f in staging.rglob('*') if f.is_file()
                            and f.name != 'COMPLETE.json'} - set(actual)
            if actual != expected or staged_extra:
                raise ValueError('MIGRATION_SHA_MISMATCH')
        marker = dict(old_marker)
        if isinstance(marker.get('files'), list):
            marker['files'] = {name: sha for name, sha in marker['files']}
        marker['migrated_from'] = str(src)
        marker['migrated_at'] = datetime.now().isoformat(timespec='seconds')
        marker['files'] = {name: sha for name, sha in manifest}
        (staging / 'COMPLETE.json').write_text(
            json.dumps(marker, ensure_ascii=False, indent=1), encoding='utf-8')
        staging.replace(dst)
        # Validate the copied, marker-complete asset BEFORE readers switch;
        # on failure the source stays authoritative and dst is removed.
        if not validate(dst):
            shutil.rmtree(dst)
            raise ValueError('MIGRATION_VALIDATE_FAILED')
        switch(dst)
        receipt = dict(src=str(src), dst=str(dst), files=len(manifest),
                       bytes=sum(f.stat().st_size for f in dst.rglob('*') if f.is_file()),
                       migrated_at=marker['migrated_at'],
                       old_copy='PRESERVED_AWAITING_AUTHORIZATION')
        return receipt
    except Exception:
        # Readers never saw a half-migrated asset: switch only ran after the
        # marker was complete, and validation failure propagates to the caller.
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        lock.unlink(missing_ok=True)
