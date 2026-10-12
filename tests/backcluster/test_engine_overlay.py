"""The patched engine is reproducible and rejects changed source/cache inputs."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster.engine_overlay import expected_identity, prepare_engine


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engine = self.root / 'llama.cpp'
        self.engine.mkdir()
        self.env = dict(os.environ, GIT_AUTHOR_NAME='Fixture', GIT_COMMITTER_NAME='Fixture',
                        GIT_AUTHOR_EMAIL='fixture@users.noreply.github.com',
                        GIT_COMMITTER_EMAIL='fixture@users.noreply.github.com')
        self.git(self.root, 'init', '-q')
        self.git(self.engine, 'init', '-q')
        (self.engine / 'CMakeLists.txt').write_text('old\n')
        (self.engine / 'build-info.cmake').write_text('protected\n')
        self.git(self.engine, 'add', '.')
        self.git(self.engine, 'commit', '-qm', 'fixture')
        self.pin = self.git(self.engine, 'rev-parse', 'HEAD').strip()
        self.git(self.root, 'add', 'llama.cpp')
        self.git(self.root, 'commit', '-qm', 'fixture pin')
        patches = self.root / 'engine-patches'
        patches.mkdir()
        self.patch = patches / '0001-fixture.patch'
        self.patch.write_text('diff --git a/CMakeLists.txt b/CMakeLists.txt\n'
                              '--- a/CMakeLists.txt\n+++ b/CMakeLists.txt\n'
                              '@@ -1 +1 @@\n-old\n+patched\n')
        self.manifest = {'schema_version': 1, 'base_commit': self.pin,
                         'patches': [{'file': self.patch.name,
                                      'sha256': hashlib.sha256(self.patch.read_bytes()).hexdigest()}]}
        self.manifest_path = patches / 'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))

    def git(self, cwd, *args):
        return subprocess.run(['git', *args], cwd=cwd, env=self.env, capture_output=True,
                              text=True, check=True).stdout

    def test_repeatable_and_concurrent_without_mutating_submodule(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: prepare_engine(self.root), range(3)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        source = results[0]['source_dir']
        self.assertEqual((source / 'CMakeLists.txt').read_text(), 'patched\n')
        self.assertEqual((self.engine / 'CMakeLists.txt').read_text(), 'old\n')
        self.assertFalse(self.git(self.engine, 'status', '--porcelain'))
        build = source / 'build-metal'
        build.mkdir()
        (build / 'result.o').write_bytes(b'output')
        self.assertEqual(prepare_engine(self.root), results[0])

    def test_patch_and_base_mismatch_fail_before_build(self):
        self.patch.write_text(self.patch.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'content mismatch'):
            prepare_engine(self.root)
        self.manifest['base_commit'] = 'a' * 40
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'base'):
            expected_identity(self.root)

    def test_dirty_base_and_changed_cached_source_fail(self):
        source = prepare_engine(self.root)['source_dir']
        (self.engine / 'CMakeLists.txt').write_text('unreviewed\n')
        with self.assertRaisesRegex(ValueError, 'clean pinned'):
            prepare_engine(self.root)
        self.git(self.engine, 'checkout', '--', 'CMakeLists.txt')
        (source / 'build-info.cmake').write_text('modified\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            prepare_engine(self.root)

    def test_unexpected_source_or_symlink_is_not_a_valid_cache(self):
        source = prepare_engine(self.root)['source_dir']
        (source / 'new.cpp').write_text('unreviewed\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            prepare_engine(self.root)
        (source / 'new.cpp').unlink()
        (source / 'redirect.cpp').symlink_to(self.engine / 'CMakeLists.txt')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            prepare_engine(self.root)

    def test_rehashed_cache_receipt_does_not_authorize_changed_source(self):
        source = prepare_engine(self.root)['source_dir']
        path = source / 'CMakeLists.txt'
        path.write_text('unreviewed\n')
        receipt_path = source / '.backcluster-source.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['source_hashes']['CMakeLists.txt'] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'source changed'):
            prepare_engine(self.root)


if __name__ == '__main__':
    unittest.main()
