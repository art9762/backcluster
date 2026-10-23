# Продолжение

Контекст владельца: MacBook Pro M4 Pro с 24 ГБ, iPhone 17 Pro Max, iPad Pro 13 M5 256 ГБ. Целевая модель **Qwen3.8-27B**, локальный coding workflow с распределёнными весами. Пользователь сейчас не имеет доступа к устройствам; программная подготовка выполняется без их аппаратной приёмки.

## Реализовано

- Pinned source provenance, отдельный BackCluster bundle/build, profiles, registry, link payload bench и conservative admission.
- Content locks, streaming contiguous GGUF sharding, global layer map, output/MTP policy, manifest/hash/partition checks.
- Content-addressed engine patch overlay с base/build identity; native middle-stage residual protocol и Mac-relayed pipeline, correctness/failure fixture без скачивания весов.
- Explicit Mac, split-prefill, split-decode/capacity, pipeline и remote-KV launch receipts; live route, exact artifact identity, memory/thermal checks; inherited inference environment не определяет скрытый режим.
- Native loaded artifact SHA-256 receipt с metadata фактически загруженной модели и invalidation после изменения файла.
- SSE bench evidence/comparison, local quality/tool fixture и OpenCode config generator.
- Отдельный BCTX portable state toolkit: три TCP-stage workers, commit/rollback/epoch/cancel checks, atomic complete checkpoints, durable SQLite tool journal, audited tool-result reconciliation через API/CLI и crash/replay tests. Процедура: [TOOL_RECOVERY.md](TOOL_RECOVERY.md).

Точный software/hardware scope и команды: [SOFTWARE_ACCEPTANCE.md](SOFTWARE_ACCEPTANCE.md), [PRE_HARDWARE.md](PRE_HARDWARE.md). Native build/test результаты проверять по актуальному CI, а не по наличию исходников тестов.

## Открыто

Нет аппаратных inference/link/memory/thermal/signing замеров на целевой тройке, quality gate реальной Qwen3.8-27B или offline OpenCode приёмки. Native distributed path требует `qwen35` metadata; проверка реального downloaded GGUF/engine support обязательна. Доступность 48 ГБ как единого memory pool не заявляется.

Capacity пока не имеет измеренного Mac RAM gain: production host использует full source GGUF, а host shard служит partition validation. BCTX checkpoints и transcript journal не подключены к production native pipeline/API agent; полное native distributed KV/GDN/RNG restore остаётся разработкой. Weighted KV, distributed speculation, compute+remote-KV combination и live thermal migration не включены.

## Следующий шаг на устройствах

Следовать [HARDWARE_RUNBOOK.md](HARDWARE_RUNBOOK.md): установить одинаковый build, сделать fresh profiles и payload tests, lock real model, Mac baseline, две пары, затем pipeline quality и memory/bench comparison. Начать со speculation off. Ошибки identity/admission исправлять в исходных данных/сборке, а не обходить checks.

Не менять engine pin случайно. Engine overlay проверяется manifest SHA-256 и content-addressed build ID; любые новые изменения требуют обновления patches и одинаковой пересборки Mac/iOS. Не переносить live tool journal назад вместе с inference checkpoint: уже выполненные side effects должны сохраняться.
