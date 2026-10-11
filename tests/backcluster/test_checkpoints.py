"""Opaque-state completeness and durable side-effect replay protection."""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster import checkpoints
from backcluster.checkpoints import CheckpointError, CheckpointStore, ToolExecutionIndeterminate, TranscriptJournal


def identity():
    return {'engine_commit': 'a' * 40, 'model_sha256': 'b' * 64,
            'tokenizer_sha256': 'c' * 64, 'template_sha256': 'd' * 64,
            'topology_sha256': 'e' * 64, 'state_format': 1, 'context': 8192,
            'hidden_size': 4, 'dtype': 'f32', 'kv_dtype': 'q8', 'protocol_version': 1}


PARTS = {'mac': b'mac-KV\0recurrent', 'ipad': b'ipad-KV\x01recurrent', 'iphone': b'iphone-KV\x02recurrent'}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'states'
        self.store = CheckpointStore(self.root)

    def save(self, **kwargs):
        values = dict(parts=PARTS, identity=identity(), session='session-1', epoch=2,
                      sequence=9, position=123, sampler_state=b'RNG-state', journal_state=b'journal')
        values.update(kwargs)
        return self.store.save(**values)

    def load(self):
        return self.store.load(identity(), PARTS)

    def rewrite_manifest(self, directory, mutate):
        path = directory / 'manifest.json'
        manifest = json.loads(path.read_bytes())
        mutate(manifest)
        data = json.dumps(manifest).encode()
        path.write_bytes(data)
        latest = json.loads((self.root / 'latest.json').read_bytes())
        latest['manifest_sha256'] = hashlib.sha256(data).hexdigest()
        (self.root / 'latest.json').write_text(json.dumps(latest))

    def test_round_trip_all_state_and_identity(self):
        path = self.save()
        loaded = self.load()
        self.assertEqual(loaded.path, path)
        self.assertEqual(loaded.parts, PARTS)
        self.assertEqual(loaded.identity, identity())
        self.assertEqual((loaded.session, loaded.epoch, loaded.sequence, loaded.position), ('session-1', 2, 9, 123))
        self.assertEqual(loaded.sampler_state, b'RNG-state')
        self.assertEqual(loaded.journal_state, b'journal')
        self.assertEqual(set(path.iterdir()), {path / ('stage-' + stage + '.bin') for stage in PARTS} |
                         {path / 'sampler.bin', path / 'journal.bin', path / 'manifest.json'})

    def test_every_identity_dimension_must_match(self):
        self.save()
        for key in identity():
            expected = identity()
            expected[key] = 'f' * len(expected[key]) if isinstance(expected[key], str) else expected[key] + 1
            with self.subTest(key=key), self.assertRaises(CheckpointError):
                self.store.load(expected, PARTS)
        expected = identity()
        expected['extra_setting'] = 'different'
        with self.assertRaises(CheckpointError):
            self.store.load(expected, PARTS)

    def test_missing_extra_and_duplicate_stages_rejected(self):
        self.save()
        for expected in (['mac', 'ipad'], ['mac', 'ipad', 'iphone', 'extra'], ['mac', 'mac', 'ipad', 'iphone']):
            with self.subTest(expected=expected), self.assertRaises(CheckpointError):
                self.store.load(identity(), expected)

    def test_same_size_corruption_and_truncation_rejected(self):
        for name in ('stage-mac.bin', 'sampler.bin', 'journal.bin'):
            for truncated in (False, True):
                directory = self.save()
                file = directory / name
                data = file.read_bytes()
                file.write_bytes(data[:-1] if truncated else b'X' + data[1:])
                with self.subTest(file=name, truncated=truncated), self.assertRaises(CheckpointError):
                    self.load()

    def test_manifest_and_pointer_corruption_rejected(self):
        directory = self.save()
        path = directory / 'manifest.json'
        data = path.read_bytes()
        path.write_bytes(data.replace(b'session-1', b'session-X'))
        with self.assertRaisesRegex(CheckpointError, 'manifest checksum'):
            self.load()
        self.save()
        pointer = self.root / 'latest.json'
        pointer.write_bytes(pointer.read_bytes()[:-1])
        with self.assertRaises(CheckpointError):
            self.load()

    def test_receipt_field_and_duplicate_json_keys_rejected(self):
        directory = self.save()
        self.rewrite_manifest(directory, lambda value: value['stages']['mac'].update(file='../outside.bin'))
        with self.assertRaisesRegex(CheckpointError, 'receipt'):
            self.load()
        self.save()
        pointer = self.root / 'latest.json'
        data = pointer.read_text()
        pointer.write_text(data[:-1] + ',"schema_version":1}')
        with self.assertRaisesRegex(CheckpointError, 'duplicate'):
            self.load()

    def test_incomplete_snapshot_is_never_latest(self):
        original_writer = checkpoints._write_file

        def failing_writer(directory, name, data):
            if name == 'stage-ipad.bin':
                self.assertFalse((self.root / 'latest.json').exists())
                raise OSError('simulated storage failure')
            original_writer(directory, name, data)

        with patch.object(checkpoints, '_write_file', failing_writer):
            with self.assertRaises(OSError):
                self.save()
        with self.assertRaises(FileNotFoundError):
            self.load()
        self.assertFalse(any(path.name.startswith('.pending-') for path in self.root.iterdir()))

    def test_previous_generation_survives_save_and_publication_failure(self):
        previous = self.save()
        for fail_on in ('stage-ipad.bin', 'latest.json'):
            original_writer = checkpoints._write_file
            original_replace = checkpoints.os.replace

            def failing_writer(directory, name, data):
                if name == fail_on:
                    raise OSError('simulated partial write')
                original_writer(directory, name, data)

            def failing_replace(source, destination, **kwargs):
                if destination == fail_on:
                    raise OSError('simulated publication failure')
                original_replace(source, destination, **kwargs)

            with self.subTest(fail_on=fail_on), patch.object(checkpoints, '_write_file', failing_writer), \
                    patch.object(checkpoints.os, 'replace', failing_replace):
                with self.assertRaises(OSError):
                    self.save(position=124)
            self.assertEqual(self.load().path, previous)
            self.assertEqual(self.load().position, 123)

    def test_referenced_missing_part_and_extra_file_rejected(self):
        directory = self.save()
        (directory / 'stage-ipad.bin').unlink()
        with self.assertRaises(CheckpointError):
            self.load()
        directory = self.save()
        (directory / 'stage-extra.bin').write_bytes(b'extra')
        with self.assertRaises(CheckpointError):
            self.load()

    def test_pending_generation_without_pointer_is_ignored(self):
        pending = self.root / '.pending-orphan'
        pending.mkdir()
        (pending / 'stage-mac.bin').write_bytes(b'partial')
        with self.assertRaises(FileNotFoundError):
            self.load()
        self.save()
        self.assertEqual(self.load().parts, PARTS)

    def test_symlinks_and_pointer_path_escape_rejected(self):
        outside = Path(self.tmp.name) / 'outside'
        outside.mkdir()
        root_link = Path(self.tmp.name) / 'root-link'
        root_link.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            CheckpointStore(root_link / 'nested')
        for name in ('stage-mac.bin', 'manifest.json', 'latest.json'):
            directory = self.save()
            target = self.root / name if name == 'latest.json' else directory / name
            copied = outside / name
            copied.write_bytes(target.read_bytes())
            target.unlink()
            target.symlink_to(copied)
            with self.subTest(name=name), self.assertRaises(CheckpointError):
                self.load()
            if name == 'latest.json':
                target.unlink()
        self.save()
        pointer = self.root / 'latest.json'
        value = json.loads(pointer.read_text())
        value['directory'] = '../outside'
        pointer.write_text(json.dumps(value))
        with self.assertRaisesRegex(CheckpointError, 'pointer'):
            self.load()

    def test_stage_directory_symlink_rejected(self):
        directory = self.save()
        moved = directory.with_name('moved')
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(CheckpointError):
            self.load()

    def test_non_regular_blob_is_rejected_without_open_blocking(self):
        directory = self.save()
        blob = directory / 'stage-mac.bin'
        blob.unlink()
        os.mkfifo(blob)
        with self.assertRaisesRegex(CheckpointError, 'regular file'):
            self.load()

    def test_blob_document_and_total_limits_on_save_and_load(self):
        for options in ({'max_blob_bytes': 3}, {'max_total_bytes': 3}, {'max_document_bytes': 100}):
            limited = CheckpointStore(self.root, **options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                limited.save(PARTS, identity(), 's', 0, 0, 0, b'rng')
        self.save()
        for options in ({'max_blob_bytes': 3}, {'max_total_bytes': 3}, {'max_document_bytes': 100}):
            with self.subTest(options=options), self.assertRaises(CheckpointError):
                CheckpointStore(self.root, **options).load(identity(), PARTS)

    def test_invalid_identity_positions_and_stage_names_rejected(self):
        for key, value in (('engine_commit', 'wrong'), ('model_sha256', 1), ('state_format', True), ('context', 0)):
            malformed = identity()
            malformed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.save(identity=malformed)
        for overrides in ({'position': 8193}, {'position': -1}, {'sequence': True}, {'epoch': -1},
                          {'parts': {'../escape': b'state'}}, {'parts': {}}, {'sampler_state': 'not bytes'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.save(**overrides)

    def test_reserved_looking_stage_names_do_not_replace_rng_or_journal(self):
        parts = {'sampler': b'stage', 'journal': b'other-stage'}
        self.save(parts=parts)
        loaded = self.store.load(identity(), parts)
        self.assertEqual(loaded.parts, parts)
        self.assertEqual(loaded.sampler_state, b'RNG-state')
        self.assertEqual(loaded.journal_state, b'journal')


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'journal.sqlite'

    def test_completed_call_returns_saved_result_across_restart(self):
        count = []

        def executor():
            count.append('executed')
            return {'stdout': 'done', 'exit_code': 0}

        with TranscriptJournal(self.path) as journal:
            first = journal.execute('call-1', 'shell', {'command': 'fixture'}, executor)
            first['stdout'] = 'changed-by-client'
            self.assertEqual(journal.execute('call-1', 'shell', {'command': 'fixture'}, executor)['stdout'], 'done')
        with TranscriptJournal(self.path) as reopened:
            self.assertEqual(reopened.execute('call-1', 'shell', {'command': 'fixture'}, executor)['stdout'], 'done')
        self.assertEqual(count, ['executed'])

    def test_same_id_different_tool_or_arguments_rejected(self):
        with TranscriptJournal(self.path) as journal:
            journal.execute('call', 'shell', {'a': 1, 'b': 2}, lambda: 'result')
            self.assertEqual(journal.execute('call', 'shell', {'b': 2, 'a': 1}, lambda: self.fail('replayed')), 'result')
            for tool, args in (('patch', {'a': 1, 'b': 2}), ('shell', {'a': 3}), ('shell', {'a': True, 'b': 2})):
                with self.subTest(tool=tool, args=args), self.assertRaisesRegex(ValueError, 'different request'):
                    journal.execute('call', tool, args, lambda: self.fail('mismatch executed'))

    def test_running_committed_before_executor_and_exception_never_replayed(self):
        with TranscriptJournal(self.path) as journal:
            def failing():
                with TranscriptJournal(self.path) as reader:
                    self.assertEqual(reader.get_call('call')['status'], 'running')
                raise RuntimeError('side effect may have happened')

            with self.assertRaisesRegex(RuntimeError, 'side effect'):
                journal.execute('call', 'patch', {}, failing)
        with TranscriptJournal(self.path) as reopened:
            with self.assertRaises(ToolExecutionIndeterminate):
                reopened.execute('call', 'patch', {}, lambda: self.fail('replayed uncertain side effect'))

    def test_real_process_death_preserves_indeterminate_call(self):
        code = '''
import os, sys
from pathlib import Path
from backcluster.checkpoints import TranscriptJournal
def side_effect():
    Path(sys.argv[2]).write_text('executed')
    os._exit(73)
with TranscriptJournal(sys.argv[1]) as journal:
    journal.execute('crashed-call', 'shell', {'command': 'fixture'}, side_effect)
'''
        marker = Path(self.tmp.name) / 'effect.txt'
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'tools'))
        result = subprocess.run([sys.executable, '-c', code, str(self.path), str(marker)], env=environment,
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 73, result.stderr.decode())
        self.assertEqual(marker.read_text(), 'executed')
        with TranscriptJournal(self.path) as journal:
            with self.assertRaises(ToolExecutionIndeterminate):
                journal.execute('crashed-call', 'shell', {'command': 'fixture'}, lambda: self.fail('replayed after crash'))

    def test_concurrent_clients_never_execute_same_callback_twice(self):
        started, release = threading.Event(), threading.Event()
        results, errors = [], []
        with TranscriptJournal(self.path) as first, TranscriptJournal(self.path) as second:
            def executor():
                started.set()
                if not release.wait(5):
                    raise RuntimeError('test timed out')
                return 'done'

            def run():
                try:
                    results.append(first.execute('call', 'shell', {}, executor))
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(started.wait(5))
                with self.assertRaises(ToolExecutionIndeterminate):
                    second.execute('call', 'shell', {}, lambda: self.fail('concurrent callback repeated'))
            finally:
                release.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results, ['done'])
            self.assertEqual(second.execute('call', 'shell', {}, lambda: self.fail('completed replay')), 'done')

    def test_unserializable_result_remains_indeterminate(self):
        with TranscriptJournal(self.path) as journal:
            with self.assertRaises(ValueError):
                journal.execute('call', 'shell', {}, lambda: object())
            with self.assertRaises(ToolExecutionIndeterminate):
                journal.execute('call', 'shell', {}, lambda: self.fail('invalid result caused replay'))

    def test_messages_and_snapshot_are_durable_and_snapshot_contains_running_calls(self):
        with TranscriptJournal(self.path) as journal:
            journal.record_message('user', 'hello')
            journal.record_message('assistant', {'content': 'world'})
            journal.execute('done', 'shell', {}, lambda: None)
            with self.assertRaises(RuntimeError):
                journal.execute('pending', 'patch', {}, lambda: (_ for _ in ()).throw(RuntimeError('failed')))
            snapshot = json.loads(journal.snapshot())
            self.assertEqual([message['content'] for message in snapshot['messages']], ['hello', {'content': 'world'}])
            self.assertEqual([call['status'] for call in snapshot['calls']], ['completed', 'running'])
        with TranscriptJournal(self.path) as reopened:
            self.assertEqual([message['role'] for message in reopened.messages()], ['user', 'assistant'])
            self.assertEqual(reopened.get_call('done')['result'], None)

    def test_json_limits_and_non_finite_values_fail_before_callback(self):
        with TranscriptJournal(self.path, max_document_bytes=128) as journal:
            for value in ('x' * 129, {'nested': float('nan')}, {1: 'non-string-key'}):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    journal.execute('call', 'shell', value, lambda: self.fail('invalid request executed'))
            self.assertIsNone(journal.get_call('call'))
            with self.assertRaises(ValueError):
                journal.record_message('user', 'x' * 129)

    def test_total_record_limit_preserves_existing_dedupe(self):
        with TranscriptJournal(self.path, max_records=2) as journal:
            journal.record_message('user', 'hello')
            journal.execute('call', 'shell', {}, lambda: 'done')
            with self.assertRaises(ValueError):
                journal.execute('other', 'shell', {}, lambda: self.fail('full journal executed'))
            with self.assertRaises(ValueError):
                journal.record_message('assistant', 'world')
            self.assertEqual(journal.execute('call', 'shell', {}, lambda: self.fail('full journal replay')), 'done')

    def test_corrupt_transcript_arguments_and_results_fail_closed(self):
        with TranscriptJournal(self.path) as journal:
            journal.record_message('user', 'hello')
            journal.execute('call', 'shell', {}, lambda: 'done')
            journal._db.execute("UPDATE messages SET content='\"other\"'")
            with self.assertRaisesRegex(ValueError, 'transcript checksum'):
                journal.messages()
            journal._db.execute("UPDATE calls SET arguments='{" + '"different":true' + "}'")
            with self.assertRaisesRegex(ValueError, 'receipt checksum'):
                journal.execute('call', 'shell', {}, lambda: self.fail('corrupt args replayed'))
            journal._db.execute("UPDATE calls SET arguments='{}', result='\"changed\"'")
            with self.assertRaisesRegex(ValueError, 'result checksum'):
                journal.execute('call', 'shell', {}, lambda: self.fail('corrupt result replayed'))

    def test_journal_and_auxiliary_symlinks_rejected(self):
        outside = Path(self.tmp.name) / 'outside'
        outside.write_bytes(b'')
        self.path.symlink_to(outside)
        with self.assertRaises((OSError, ValueError)):
            TranscriptJournal(self.path)
        self.path.unlink()
        Path(str(self.path) + '-wal').symlink_to(outside)
        with self.assertRaises((OSError, ValueError)):
            TranscriptJournal(self.path)

    def test_unknown_journal_version_rejected(self):
        with sqlite3.connect(str(self.path)) as database:
            database.execute('PRAGMA user_version=99')
        with self.assertRaisesRegex(ValueError, 'version'):
            TranscriptJournal(self.path)


if __name__ == '__main__':
    unittest.main()
