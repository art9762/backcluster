# Backlog

BC-001 **IN PROGRESS**: CLI `doctor/profile`, схема снимков, read-only mobile report и socket tests реализованы; Apple build, измерения на трёх устройствах и полный compatibility handshake не выполнены. BC-002/003/010/011/030 подготовлены программно частично; их аппаратная приёмка и runtime integration открыты. Остальные задачи ниже **TODO**. См. [PRE_HARDWARE.md](PRE_HARDWARE.md). ID предназначены для issues; это Markdown-план. `P0` — обязательная основа, `P1` — основной результат, `P2` — оптимизация.

| ID | Приоритет | Зависит от | Работа и точки изменения | Проверяемый результат |
|---|---|---|---|---|
| BC-001 | P0 | — | **IN PROGRESS:** `backcluster doctor/profile`, `tools/backcluster/`, RPC `profile`; [инструкция](DIAGNOSTICS.md) | Код и tests готовы; full Apple archive в CI; приёмка трёх узлов ожидает железо |
| BC-002 | P0 | 001 | Cable payload bench, mobile signing и standalone app | RTT/throughput для обоих links, app budget без debugger, две независимые связи |
| BC-003 | P0 | — | Model/draft/shard lock и GGUF metadata validation | Pinned revision, bytes/hash, template, layer map; mismatch блокируется |
| BC-004 | P0 | 001,003 | Mac-only baseline, `serve.sh`, `bench/` | IQ4, 16k/32k/64k где помещается; одинаковые настройки, пять повторов |
| BC-005 | P0 | 002,004 | Mac+iPhone и Mac+iPad; split search | Лучший pair/split по wall time, проверено влияние SME и draft |
| BC-006 | P0 | 004 | OpenCode smoke и quality fixtures | SSE, tool round-trip, reasoning parsing, read/edit/test loop |
| BC-010 | P1 | 001,002 | Registry вместо выбора первого устройства | Stable local mapping device-interface-endpoint-role; identity checks |
| BC-011 | P1 | 010,003 | Per-device admission, убрать общие heuristics `serve.sh` | Каждому worker свой budget/reserve; неизвестный лимит не принимается |
| BC-012 | P1 | 005,011 | Два настоящих remote-KV workers, существующий round-robin | Consistent output, подтверждён KV на обоих узлах, сохранены raw rows |
| BC-013 | P1 | 012,020 | Weighted page ownership, `ggml-metal-remote.mm` | Правильный position mapping, fetch/truncate/restore, heterogenous budgets |
| BC-020 | P0 | 003 | Создать downstream engine fork; обновить `.gitmodules` только после публикации pin | Clean recursive clone получает нужный commit; upstream remote сохранён |
| BC-021 | P1 | 005,011 | Двухузловой split-decode Mac+iPad, IQ4 затем Q5 | RAM Mac -4 GiB на сопоставимом профиле; нет скрытого чтения tail weights |
| BC-022 | P1 | 020,021 | General shard splitter `[start,end)`, global layer mapping | Диапазоны покрывают trunk ровно один раз; tensors/arrays/MTP/head валидны |
| BC-023 | P1 | 022 | Middle-stage protocol в engine tail-server и iOS bridge | Возвращает residual; bounded framing, identity/epoch checks, ошибки совместимости |
| BC-024 | P1 | 023 | Multi-stage coordinator в `llama-split.cpp` | Mac+iPad+iPhone prefill/decode без speculation, trace и correctness gate |
| BC-025 | P2 | 012,024 | Совместить compute-shards и remote KV | Tail видит всю нужную историю; нет двойного учёта или пропуска attention |
| BC-026 | P1 | 024 | Distributed checkpoints, transcript journal, cache identity | Atomic complete snapshot; restore/cancel/reconnect; tools не исполняются повторно |
| BC-027 | P2 | 026 | Distributed speculative decode/recurrent replay | Accept/reject откатывает все стадии; measured accepted-token gain |
| BC-028 | P1 | 011,024 | Thermal policy и безопасная смена профиля | Warning/serious/critical state обрабатываются без порчи сессии |
| BC-030 | P1 | 012,021 | `backcluster` CLI, installer, bundle ID, release naming | Команды doctor/profile/plan/serve/bench; upstream install/update не вызываются по ошибке |
| BC-031 | P1 | 006,030 | Профиль OpenCode по установленной версии | Одна локальная модель; tools, reasoning, context limits, cancel проверены |
| BC-032 | P1 | 024,026,028,031 | Offline acceptance и release report | 10 задач, час нагрузки, fail tests, воспроизводимая установка и честный отчёт |

## Первые небольшие изменения после подключения железа

1. BC-001/002: сохранить обезличенный профиль и raw link samples. Не менять kernels.
2. BC-003/004: artifact lock и baseline с существующим engine.
3. BC-005/006: выбрать pair и получить первый полезный coding loop.
4. BC-010/011: заменить эвристики wrapper на явные бюджеты и роли.
5. BC-021: проверить реальное освобождение RAM до реализации трёх стадий.

Одна работа меняет один фактор. Каждый performance change сопровождается одинаковой моделью, настройками, командами, до/после и quality check. Новый protocol требует совместимой сборки приложения. Код engine меняется в его repository, не только внутри локального submodule.

## Что не делать в начале

Не тратить первый этап на новый inference framework, свой редактор кода, облачный dashboard, обучение drafter, модели на сотни миллиардов параметров или background GPU workaround. Не фиксировать пропорции слоёв по одной только RAM. Не выдавать вспомогательный независимый LLM-agent на телефоне за distributed inference основной модели.

## Подготовлено до аппаратной границы, 2026-10-10

- BC-002: native control-port payload echo + CLI raw/summary + concurrent workers. Signing, actual route/link mode и standalone budgets ещё не измерены.
- BC-003: local artifact content lock, GGUF metadata/layout/sha verification. Реальная модель, drafter/shard provenance и engine support не приняты.
- BC-010: explicit private registry, live route и product/protocol/engine checks. Не является hardware authentication и пока не управляет upstream inference.
- BC-011: conservative independent memory admission estimates, unknown/thermal rejection. Context state needs и live launch admission требуют измерений.
- BC-030: versioned CLI, separate BackCluster app build и safe Mac-only launcher. Multi-worker product launch и offline installer ждут pair baseline.
- BC-004/006: bounded local SSE benchmark подготовлен. Качество, tool round-trip, OpenCode config и cancellation на целевом runtime ещё не приняты.
