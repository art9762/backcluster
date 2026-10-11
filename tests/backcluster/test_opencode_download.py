"""The opt-in client gate refuses corrupted archives before executable use."""
import base64
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('check_opencode', ROOT / 'scripts/check-opencode.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def archive(link=False, version=gate.VERSION):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as tar:
        metadata = json.dumps({'name': gate.PACKAGE, 'version': version}).encode()
        member = tarfile.TarInfo('package/package.json'); member.size = len(metadata)
        tar.addfile(member, io.BytesIO(metadata))
        member = tarfile.TarInfo('package/bin/opencode')
        if link:
            member.type = tarfile.SYMTYPE; member.linkname = '../../outside'
            tar.addfile(member)
        else:
            data = b'fixture binary bytes'; member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return output.getvalue()


def integrity(data):
    return 'sha512-' + base64.b64encode(hashlib.sha512(data).digest()).decode()


class ClientDownloadTests(unittest.TestCase):
    def test_unverified_bytes_never_become_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                gate.download_binary(directory, lambda *a, **k: io.BytesIO(archive()))
            self.assertFalse((directory / 'opencode').exists())

    def test_verified_fixture_is_extracted_by_exact_member_only(self):
        data = archive()
        with tempfile.TemporaryDirectory() as tmp, patch.object(gate, 'INTEGRITY', integrity(data)):
            executable, provenance = gate.download_binary(Path(tmp), lambda *a, **k: io.BytesIO(data))
            self.assertEqual(executable.read_bytes(), b'fixture binary bytes')
            self.assertEqual(executable.stat().st_mode & 0o777, 0o500)
            self.assertEqual(provenance['binary_sha256'], hashlib.sha256(b'fixture binary bytes').hexdigest())
            self.assertFalse((Path(tmp) / 'package').exists())

    def test_identity_and_symlinks_are_rejected_even_after_integrity(self):
        for data in (archive(link=True), archive(version='other')):
            with self.subTest(), tempfile.TemporaryDirectory() as tmp, patch.object(gate, 'INTEGRITY', integrity(data)):
                with self.assertRaises(ValueError): gate.download_binary(Path(tmp), lambda *a, **k: io.BytesIO(data))
                self.assertFalse((Path(tmp) / 'opencode').exists())

    def test_size_bound_applies_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(gate, 'MAX_ARCHIVE', 10):
            with self.assertRaisesRegex(ValueError, 'byte or time limit'):
                gate.download_binary(Path(tmp), lambda *a, **k: io.BytesIO(archive()))
            self.assertFalse((Path(tmp) / 'opencode').exists())


if __name__ == '__main__': unittest.main()
