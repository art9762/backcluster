"""Actual transport, replay, identity, partial commit and cancellation tests."""
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from backcluster.runtime import (Coordinator, FRAME, Identity, MAGIC, MAX_METADATA, MAX_PAYLOAD,
                                PROTOCOL_VERSION, ReferenceStage, RuntimeErrorCode, SocketStage,
                                Stage, StageWorker, WorkerServer, pack_f32, receive_frame,
                                runtime_selftest, send_frame, topology_identity, unpack_f32)


def fixture():
    stages = [Stage(alias, i, i + 1, str(i + 1) * 64) for i, alias in enumerate(('mac', 'ipad', 'iphone'))]
    identity = Identity('a' * 40, 'b' * 64, 'c' * 64, 'd' * 64, topology_identity(stages, 3), 1, 16, 2)
    adapters = [ReferenceStage(i + 1) for i in range(3)]
    workers = [StageWorker(identity, stage, adapter) for stage, adapter in zip(stages, adapters)]
    return identity, stages, adapters, workers


class RuntimeTests(unittest.TestCase):
    def test_three_stage_socket_lifecycle(self):
        receipt = runtime_selftest()
        self.assertTrue(receipt['ok'], receipt['checks'])
        self.assertTrue(receipt['software_only'])
        self.assertFalse(receipt['inference_validated'])
        self.assertFalse(receipt['hardware_validated'])

    def test_gap_overlap_duplicate_alias_and_order_rejected(self):
        _, stages, _, _ = fixture()
        for bad in ([stages[0], stages[2]], list(reversed(stages)),
                    [stages[0], Stage('mac', 1, 3, 'f' * 64)], [stages[0], stages[1]]):
            with self.assertRaisesRegex(RuntimeErrorCode, 'invalid_topology'):
                topology_identity(bad, 3)

    def test_all_identity_checks_precede_any_begin(self):
        identity, stages, adapters, workers = fixture()
        other = asdict(identity); other['template_sha256'] = 'f' * 64
        workers[-1].identity = Identity(**other)
        coordinator = Coordinator(identity, stages, workers, 3)
        with self.assertRaisesRegex(RuntimeErrorCode, 'compatibility_mismatch'):
            coordinator.begin('test')
        self.assertTrue(all(worker.session is None for worker in workers))

    def test_identity_boolean_version_rejected(self):
        identity, stages, _, workers = fixture()
        for key in ('protocol_version', 'state_format'):
            altered = asdict(identity); altered[key] = True
            with self.assertRaisesRegex(RuntimeErrorCode, 'compatibility_mismatch'):
                workers[0].call({'op': 'hello', 'identity': altered, 'stage': asdict(stages[0])})

    def test_retry_collision_and_cancel_restore_pending(self):
        identity, stages, adapters, workers = fixture()
        worker = workers[0]
        worker.call({'op': 'begin', 'session': 'test', 'epoch': 1})
        meta = dict(op='forward', session='test', epoch=1, sequence=0, position=0, tokens=1)
        first = worker.call(meta, pack_f32([2, 3]))
        self.assertEqual(worker.call(meta, pack_f32([2, 3])), first)
        self.assertEqual(adapters[0].calls, 1)
        with self.assertRaisesRegex(RuntimeErrorCode, 'sequence_collision'):
            worker.call(meta, pack_f32([3, 3]))
        with self.assertRaisesRegex(RuntimeErrorCode, 'uncommitted_state'):
            worker.call(dict(op='snapshot', session='test', epoch=1))
        result, _ = worker.call(dict(op='cancel', session='test', epoch=1))
        self.assertEqual(result['position'], 0)
        self.assertEqual(adapters[0].total_tokens, 0)
        with self.assertRaisesRegex(RuntimeErrorCode, 'session_cancelled'):
            worker.call(meta, pack_f32([2, 3]))
        with self.assertRaisesRegex(RuntimeErrorCode, 'invalid_epoch'):
            worker.call(dict(op='begin', session='test', epoch=1))
        worker.call(dict(op='begin', session='test', epoch=2))
        with self.assertRaisesRegex(RuntimeErrorCode, 'session_epoch_mismatch'):
            worker.call(meta, pack_f32([2, 3]))

    def test_compute_failure_rolls_back_recurrent_state(self):
        class BrokenStage(ReferenceStage):
            def apply(self, payload, tokens, position):
                self.total_tokens += tokens
                raise OSError('private detail')
        identity, stages, _, workers = fixture()
        broken = BrokenStage(2)
        workers[1].adapter = broken
        coordinator = Coordinator(identity, stages, workers, 3)
        coordinator.begin('test')
        with self.assertRaisesRegex(RuntimeErrorCode, '^distributed_step_failed$'):
            coordinator.forward(pack_f32([1, 2]), 1)
        self.assertFalse(coordinator.poisoned)
        self.assertEqual(broken.total_tokens, 0)
        self.assertTrue(all(worker.position == 0 and worker.pending is None for worker in workers))

    def test_stale_rollback_cannot_discard_a_new_pending_transaction(self):
        _, _, adapters, workers = fixture()
        worker = workers[0]
        worker.call(dict(op='begin', session='test', epoch=1))
        common = dict(session='test', epoch=1)
        worker.call(dict(common, op='forward', sequence=0, position=0, tokens=1), pack_f32([1, 2]))
        worker.call(dict(common, op='commit', sequence=0))
        worker.call(dict(common, op='forward', sequence=1, position=1, tokens=1), pack_f32([1, 2]))
        with self.assertRaisesRegex(RuntimeErrorCode, 'missing_transaction'):
            worker.call(dict(common, op='rollback', sequence=0))
        self.assertEqual(adapters[0].total_tokens, 2)
        worker.call(dict(common, op='rollback', sequence=1))
        self.assertEqual(adapters[0].total_tokens, 1)

    def test_completed_distributed_commit_rejects_late_rollback(self):
        identity, stages, adapters, workers = fixture()
        coordinator = Coordinator(identity, stages, workers, 3)
        coordinator.begin('test')
        coordinator.forward(pack_f32([1, 2]), 1)
        for worker in workers:
            with self.assertRaisesRegex(RuntimeErrorCode, 'transaction_already_sealed'):
                worker.call(dict(op='rollback', session='test', epoch=1, sequence=0))
        self.assertTrue(all(adapter.total_tokens == 1 for adapter in adapters))

    def test_partial_seal_requires_complete_checkpoint_restore(self):
        from backcluster.checkpoints import CheckpointStore
        identity, stages, _, workers = fixture()
        class FailingSeal:
            def __init__(self, worker):
                self.worker, self.fail = worker, True
            def call(self, meta, payload=b''):
                if self.fail and meta['op'] == 'seal':
                    self.fail = False
                    raise RuntimeErrorCode('transport_failed')
                return self.worker.call(meta, payload)
        coordinator = Coordinator(identity, stages, [workers[0], FailingSeal(workers[1]), workers[2]], 3)
        coordinator.begin('test')
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            coordinator.checkpoint(store, b'rng')
            with self.assertRaisesRegex(RuntimeErrorCode, 'distributed_commit_unconfirmed'):
                coordinator.forward(pack_f32([1, 2]), 1)
            self.assertTrue(coordinator.poisoned)
            self.assertEqual(coordinator.position, 1)
            coordinator.restore(store)
            self.assertTrue(all(worker.position == 0 for worker in workers))

    def test_unknown_worker_state_poisoned_and_full_checkpoint_recovers(self):
        from backcluster.checkpoints import CheckpointStore
        identity, stages, _, workers = fixture()
        class DisappearingWorker:
            def __init__(self, worker):
                self.worker, self.disconnected = worker, False
            def call(self, meta, payload=b''):
                if self.disconnected:
                    raise RuntimeErrorCode('transport_failed')
                result = self.worker.call(meta, payload)
                if meta['op'] == 'forward':
                    self.disconnected = True
                    raise RuntimeErrorCode('transport_failed')
                return result
        lost = DisappearingWorker(workers[1])
        coordinator = Coordinator(identity, stages, [workers[0], lost, workers[2]], 3)
        coordinator.begin('test')
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            coordinator.checkpoint(store, b'rng')
            with self.assertRaisesRegex(RuntimeErrorCode, 'distributed_state_uncertain'):
                coordinator.forward(pack_f32([1, 2]), 1)
            self.assertTrue(coordinator.poisoned)
            with self.assertRaisesRegex(RuntimeErrorCode, 'coordinator_unavailable'):
                coordinator.forward(pack_f32([1, 2]), 1)
            lost.disconnected = False
            coordinator.restore(store)
            self.assertEqual(coordinator.epoch, 2)
            self.assertFalse(coordinator.poisoned)
            self.assertTrue(all(worker.position == 0 and worker.pending is None for worker in workers))

    def test_partial_restore_can_retry_with_new_common_epoch(self):
        from backcluster.checkpoints import CheckpointStore
        identity, stages, _, workers = fixture()
        class FailingRestore:
            def __init__(self, worker):
                self.worker, self.fail = worker, True
            def call(self, meta, payload=b''):
                if self.fail and meta['op'] == 'restore':
                    self.fail = False
                    raise RuntimeErrorCode('transport_failed')
                return self.worker.call(meta, payload)
        coordinator = Coordinator(identity, stages, [workers[0], FailingRestore(workers[1]), workers[2]], 3)
        coordinator.begin('test')
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            coordinator.checkpoint(store, b'rng')
            coordinator.forward(pack_f32([1, 2]), 1)
            with self.assertRaisesRegex(RuntimeErrorCode, 'checkpoint_restore_incomplete'):
                coordinator.restore(store)
            self.assertTrue(coordinator.poisoned)
            coordinator.restore(store)
            self.assertEqual(coordinator.epoch, 3)
            self.assertTrue(all(worker.epoch == 3 and worker.position == 0 for worker in workers))

    def test_activation_shape_nonfinite_and_context_gate(self):
        identity, stages, _, workers = fixture()
        coordinator = Coordinator(identity, stages, workers, 3)
        coordinator.begin('test')
        for payload, tokens in ((b'123', 1), (struct.pack('<ff', float('nan'), 1), 1), (b'', 0), (pack_f32([1, 2]), True)):
            with self.assertRaisesRegex(RuntimeErrorCode, 'invalid_activation'):
                coordinator.forward(payload, tokens)
        coordinator.forward(pack_f32(range(32)), 16)
        with self.assertRaisesRegex(RuntimeErrorCode, 'distributed_step_failed'):
            coordinator.forward(pack_f32([1, 2]), 1)
        self.assertEqual(coordinator.position, 16)


class FramingTests(unittest.TestCase):
    def test_fragmented_socket_frame(self):
        left, right = socket.socketpair()
        with left, right:
            metadata = b'{"op":"status"}'
            wire = FRAME.pack(MAGIC, PROTOCOL_VERSION, len(metadata), 4) + metadata + b'data'
            def sender():
                for byte in wire:
                    left.sendall(bytes([byte]))
            thread = threading.Thread(target=sender)
            thread.start()
            self.assertEqual(receive_frame(right, time.monotonic() + 2), ({'op': 'status'}, b'data'))
            thread.join()

    def test_oversized_length_rejected_before_allocation(self):
        for metadata_size, payload_size in ((MAX_METADATA + 1, 0), (0, MAX_PAYLOAD + 1)):
            left, right = socket.socketpair()
            with left, right:
                left.sendall(FRAME.pack(MAGIC, PROTOCOL_VERSION, metadata_size, payload_size))
                with self.assertRaisesRegex(RuntimeErrorCode, 'frame_too_large'):
                    receive_frame(right, time.monotonic() + 1)

    def test_deadline_applies_across_partial_receives(self):
        left, right = socket.socketpair()
        with left, right:
            left.sendall(b'B')
            start = time.monotonic()
            with self.assertRaisesRegex(RuntimeErrorCode, 'timeout'):
                receive_frame(right, start + .05)
            self.assertLess(time.monotonic() - start, .5)

    def test_duplicate_nonfinite_and_truncated_json(self):
        for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{', b'[]'):
            left, right = socket.socketpair()
            with left, right:
                left.sendall(FRAME.pack(MAGIC, PROTOCOL_VERSION, len(data), 0) + data)
                with self.assertRaises(RuntimeErrorCode):
                    receive_frame(right, time.monotonic() + 1)

    def test_harness_refuses_remote_endpoint(self):
        for host in ('0.0.0.0', '169.254.1.2', 'example.com'):  # sensitive-ok: synthetic rejected endpoint
            with self.assertRaisesRegex(RuntimeErrorCode, 'loopback_endpoint_required'):
                SocketStage(host, 50060)


if __name__ == '__main__':
    unittest.main()
