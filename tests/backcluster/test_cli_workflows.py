"""Public commands connect the independently tested preparation modules."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster.cli import main
from backcluster.artifacts import make_lock
from backcluster.devices import engine_status
from backcluster.checkpoints import TranscriptJournal
from generate_tiny_model import generate


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.source = self.directory / 'source.gguf'
        generate(self.source)
        self.lock = self.directory / 'lock.json'
        self.lock.write_text(json.dumps(make_lock(self.source, 'fixture/model', 'a' * 40,
                                                  engine_status()['expected_commit'])))

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(map(str, args)))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_public_shard_partition_and_corruption_workflow(self):
        manifests = []
        for role, start, end in [('host', 0, 2), ('middle', 2, 6), ('tail', 6, 8)]:
            shard = self.directory / (role + '.gguf')
            manifest = self.directory / (role + '.json')
            code, out, err = self.invoke('shard', '--file', self.source, '--lock', self.lock,
                                         '--start', start, '--end', end, '--role', role,
                                         '--output', shard, '--manifest', manifest)
            self.assertEqual((code, out, err), (0, '', ''))
            manifests += ['--manifest', manifest]
            code, out, err = self.invoke('shard-check', '--file', shard, '--manifest', manifest,
                                         '--source', self.source)
            self.assertEqual(code, 0, err)
            self.assertTrue(json.loads(out)['ok'])
        code, out, err = self.invoke('partition-check', *manifests)
        self.assertEqual(code, 0, err)
        self.assertEqual([s['role'] for s in json.loads(out)['shards']], ['host', 'middle', 'tail'])
        with shard.open('r+b') as handle:
            handle.seek(-1, 2)
            handle.write(b'!')
        code, out, err = self.invoke('shard-check', '--file', shard, '--manifest', manifest)
        self.assertEqual(code, 1)
        self.assertNotIn('Traceback', err)

    def test_manifest_path_cannot_overwrite_source_or_shard(self):
        initial = self.source.read_bytes()
        code, out, err = self.invoke('shard', '--file', self.source, '--lock', self.lock,
                                     '--start', 0, '--end', 2, '--role', 'host',
                                     '--output', self.directory / 'host.gguf', '--manifest', self.source)
        self.assertEqual(code, 1)
        self.assertEqual(self.source.read_bytes(), initial)

    def test_report_and_config_destinations_cannot_destroy_model_inputs(self):
        initial = self.source.read_bytes()
        code, out, err = self.invoke('model', '--file', self.source, '--repository', 'fixture/model',
                                     '--revision', 'a' * 40, '--output', self.source)
        self.assertEqual(code, 1)
        self.assertEqual(self.source.read_bytes(), initial)
        original_lock = self.lock.read_bytes()
        code, out, err = self.invoke('opencode-config', '--lock', self.lock, '--context', 512,
                                     '--tokens', 128, '--client-version', '1.2.3', '--output', self.lock)
        self.assertEqual(code, 1)
        self.assertEqual(self.lock.read_bytes(), original_lock)

    def test_opencode_output_is_directly_loadable_client_config(self):
        config = self.directory / 'opencode.json'
        code, out, err = self.invoke('opencode-config', '--lock', self.lock, '--context', 512,
                                     '--tokens', 128, '--client-version', '1.2.3', '--output', config)
        self.assertEqual((code, out, err), (0, '', ''))
        parsed = json.loads(config.read_text())
        self.assertEqual(parsed['model'], parsed['small_model'])
        self.assertEqual(parsed['provider']['backcluster']['options']['baseURL'], 'http://127.0.0.1:8080/v1')
        code, out, err = self.invoke('opencode-config', '--lock', self.lock, '--context', 512,
                                     '--tokens', 128, '--client-version', '2.0.0', '--output', config)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(config.read_text()), parsed)

    def test_journal_inspect_reconcile_and_replay_workflow(self):
        database = self.directory / 'journal.sqlite'
        request = self.directory / 'reconcile.json'
        with TranscriptJournal(database) as journal:
            with self.assertRaises(RuntimeError):
                journal.execute('fixture-call', 'edit', {'path': 'fixture.py'},
                                lambda: (_ for _ in ()).throw(RuntimeError('after side effect')))
        code, out, err = self.invoke('journal-call', '--journal', database, '--call-id', 'fixture-call')
        self.assertEqual(code, 0, err)
        call = json.loads(out)['call']
        self.assertEqual(call['status'], 'running')
        request.write_text(json.dumps({'call_id': call['call_id'], 'tool': call['tool'], 'arguments': call['arguments'],
                                       'result': {'edited': True}, 'evidence': 'Inspected fixture.py and verified the edit'}))
        for _ in range(2):
            code, out, err = self.invoke('journal-reconcile', '--journal', database, '--request', request)
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out)['call']['status'], 'completed')
        with TranscriptJournal(database) as journal:
            self.assertEqual(journal.execute('fixture-call', 'edit', {'path': 'fixture.py'},
                                             lambda: self.fail('CLI recovery replayed side effect')), {'edited': True})

    def test_journal_cli_rejects_missing_database_and_malformed_request(self):
        database = self.directory / 'missing.sqlite'
        code, out, err = self.invoke('journal-call', '--journal', database, '--call-id', 'call')
        self.assertEqual(code, 1)
        self.assertFalse(database.exists())
        request = self.directory / 'request.json'
        for text in ('{"call_id":"first","call_id":"second"}', '{"call_id":"missing fields"}'):
            request.write_text(text)
            code, out, err = self.invoke('journal-reconcile', '--journal', database, '--request', request)
            self.assertEqual(code, 1)
            self.assertNotIn('Traceback', err)
            self.assertFalse(database.exists())
        database.write_text('not a sqlite database')
        code, out, err = self.invoke('journal-call', '--journal', database, '--call-id', 'call')
        self.assertEqual(code, 1)
        self.assertIn('database operation failed', err)
        self.assertNotIn('Traceback', err)

    def test_journal_output_cannot_overwrite_database_auxiliaries_or_request(self):
        database = self.directory / 'journal.sqlite'
        request = self.directory / 'request.json'
        request.write_text('{}')
        with TranscriptJournal(database) as journal:
            journal.execute('call', 'shell', {}, lambda: 'done')
            for suffix in ('', '-wal', '-shm', '-journal', '.executors.lock'):
                with self.subTest(suffix=suffix):
                    code, out, err = self.invoke('journal-call', '--journal', database, '--call-id', 'call',
                                                 '--output', str(database) + suffix)
                    self.assertEqual(code, 1)
                    self.assertIn('distinct from inputs', err)
            code, out, err = self.invoke('journal-reconcile', '--journal', database, '--request', request, '--output', request)
            self.assertEqual(code, 1)
            self.assertEqual(request.read_text(), '{}')
            self.assertEqual(journal.get_call('call')['result'], 'done')


if __name__ == '__main__':
    unittest.main()
