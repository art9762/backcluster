# Программная приёмка и аппаратная граница

Эта версия содержит исполняемый код подготовки и распределённого запуска. Аппаратная приёмка Mac/iPad/iPhone, реальные веса Qwen3.8-27B и измеренная производительность остаются отдельными проверками. Успешные тесты с маленькой моделью не подтверждают качество ответов целевой модели или доступный объём памяти на iOS.

## Проверки без устройств

Из чистого recursive clone:

```bash
git submodule update --init --recursive
scripts/check.sh
scripts/test-native-pipeline.sh
./backcluster runtime-selftest --output .backcluster/runtime-selftest.json
```

`scripts/check.sh` запускает Python suite, shell syntax, listener/privacy checks и native payload-link test. Python tests включают реальные локальные TCP/SSE-соединения, malformed/oversized/truncated frames, timeout, cancellation, неправильные identity, изменённые GGUF, память/thermal admission, raw benchmark comparison и tool round-trip. Журнал проверяется также реальным аварийным завершением subprocess после side effect.

`scripts/test-native-pipeline.sh` собирает CPU overlay engine и генерирует маленькую untrained Qwen35 GGUF без скачивания весов. Native gate проверяет logits full-model против цепочки, prompt/decode/appends/reset, f32/f16 residual, identity/range/epoch/framing rejection и bounded failure при остановке каждого worker. Результат этого gate нужно проверять в текущем запуске/CI, а не по наличию теста.

`runtime-selftest` поднимает три TCP workers на loopback и проверяет последовательный forward, повтор sequence без повторного продвижения состояния, потерянное commit acknowledgment, rollback всех стадий, полный checkpoint replay и общую отмену. Его receipt явно содержит `software_only=true`, `inference_validated=false`, `hardware_validated=false`, `adapter=deterministic_reference_arithmetic`.

## Два разных runtime-контракта

| Компонент | Что реализовано | Что подтверждает проверка |
|---|---|---|
| Native SPT stage extension в engine | Middle residual, source/range identity, epoch/request IDs, bounded messages, Mac-relayed chain и worker failure handling | Настоящие вызовы `llama_decode` на deterministic CPU fixture; Apple backend и целевая модель проверяются отдельно |
| Python BCTX toolkit | `StageWorker`, transactional `Coordinator`, socket transport, полный checkpoint всех opaque adapter states | Software lifecycle adapters, реализующих полный snapshot/restore; это отдельный wire protocol |
| `CheckpointStore` | Atomic complete manifest, stage bytes, sampler bytes, identity и SHA-256 каждого файла | Неполное/испорченное/чужое состояние не читается как полный checkpoint |
| `TranscriptJournal` | SQLite `running` до tool callback, сохранённый результат, запрет автоматического повторения после crash | Tool call с тем же ID/arguments не выполняется второй раз; неопределённый результат требует reconciliation |

Native fixture содержит GDN/recurrent и full-attention layers, настоящие GGUF-тензоры и маленький vocabulary. Она предназначена для сравнения monolithic и distributed математики, а не для оценки языка.

Python checkpoint toolkit пока **не подключён к state ownership production `llama-server`/SPT pipeline**. Его replay не доказывает сохранение всех native KV/GDN shards или sampler RNG. Native stage restore primitive также не означает готовый distributed checkpoint launcher. Нельзя восстанавливать live tool journal из более старого inference checkpoint: уже выполненные инструменты должны остаться в журнале.

## Source и build identity

Engine gitlink остаётся закреплённым. Новые engine changes хранятся в `engine-patches/`; `manifest.json` фиксирует base commit и SHA-256 каждого patch. Materializer создаёт отдельное проверяемое дерево в `.backcluster/engine-source/<build_id>` и отвергает изменённый cache. `engine_commit` идентифицирует base, а `engine_build_id` — точный набор patches. Оба значения должны совпадать у Mac и worker.

```bash
PYTHONPATH=tools python3 -m backcluster.engine_overlay --build-id
./backcluster build
IPA=1 JOBS=3 scripts/build-backcluster.sh
```

Apple CI включает Objective-C++ syntax, полный Mac server build, настоящий Metal matrix graph и archive приложения. `scripts/test-metal.sh` требует рабочую shader library и проверяет 16 результатов F32 matmul через явно созданный Metal backend, затем повторяет с выключенным BF16 для older-device path. Код 0 от `--help` не считается подтверждением рабочей GPU library. На Linux нет Apple SDK; успешные Linux checks не заменяют эти gates. Generated header при ручной сборке обязателен для strict admission: `unknown` не принимается как проверенная identity.

## Shards и launch admission

`shard` создаёт диапазоны `[start,end)`, перенумеровывает block tensors, сохраняет global layer map, преобразует layer arrays, разделяет output/MTP policy и записывает полный source hash. `shard-check` сверяет GGUF с manifest; `partition-check` требует ровного покрытия trunk без overlap/gap и сохранённого output head. Границы должны сохранять full-attention interval. Lock/manifest проверяют bytes и связь shard с parent; заявленный publisher не удостоверяется. Поддержка произвольной архитектуры движком не доказывается.

Исполняемый cluster launcher поддерживает `split-prefill`, `split-decode`/`capacity`, `pipeline` и отдельный `remote-kv`. Native distributed path сейчас ограничен архитектурой `qwen35` из metadata; похожее маркетинговое название модели не заменяет это требование. Pipeline требует exactly one `middle_stage` и one `tail_stage`; выбор не зависит от порядка discovery.

Dry-run выдаёт проверяемый launch receipt с explicit host/worker budgets, reserve, peak state estimates, shards, context/cache settings и разрешёнными environment variables. Live launch заново проверяет реальный Mac route/interface, worker profile, protocol/base/build identity, доступную память и thermal state. Снимок не разрешает обойти live checks.

Native `loaded_tail` формируется при eager model load: SHA-256 файла до load, fstat/lstat consistency после load и metadata фактически загруженного `llama_model`. Receipt содержит source/artifact hashes, range, output head, bytes, model context и build identity; путь телефона не публикуется. Изменение/замена файла инвалидирует receipt. Legacy tail может работать в прежнем режиме, но без полного receipt strict cluster launcher его отвергает.

## Реальный OpenCode client без аппаратной модели

```bash
# Отдельный network opt-in: официальный Linux x64 baseline OpenCode 1.19.0.
python3 scripts/check-opencode.py
# Offline после установки клиента той же версии; подходит также для Mac.
python3 scripts/check-opencode.py --binary /absolute/path/to/opencode
# Только tests с уже доступным клиентом:
BACKCLUSTER_OPENCODE_BIN=/absolute/path/to/opencode \
  python3 -m unittest discover -s tests/backcluster -p test_quality.py -v
```

Helper проверяет закреплённый SHA512 SRI официального npm tarball до extraction/execution, ограничивает размеры и использует temporary directory без global install. Pin, source URL и integrity находятся в [scripts/check-opencode.py](../../scripts/check-opencode.py). При `--binary` receipt фиксирует SHA-256 предоставленного executable и проверяет точную версию. Настройки обновлений/model fetch/plugins отключены; клиент получает отдельный пустой HOME и окружение без credentials. Обычный `scripts/check.sh` остаётся offline; два installed-client tests явно пропускаются, если `BACKCLUSTER_OPENCODE_BIN` не задан. CI job `opencode-client` запускает этот отдельный gate.

Проверяется настоящий OpenCode `debug config`, локальный provider/model alias, SSE, отдельный reasoning и настоящий builtin `read → edit → bash` loop: клиент меняет fixture-файл и выполняет три regression tests в tempdir. Решения fake OpenAI backend заранее зафиксированы; эта проверка подтверждает software integration клиента, а не качество целевой модели. В текущем software check OpenCode 1.19.0 прошёл оба installed-client tests. Receipt `.backcluster/opencode-client-smoke.json` содержит client provenance, hashes исходников/test output, exit code и `hardware_validated=false`, `model_inference_validated=false`.

## Что остаётся проверить и разработать

- На целевом железе: установка/signing без debugger, два кабельных маршрута, payload bandwidth, real-model logits/качество, состояние после reset/cancel/disconnect, peak memory, swap, thermal и длительный coding loop.
- Настоящий выигрыш capacity по RAM ещё не измерен. Host использует full source GGUF; tensor override переносит удалённые блоки с GPU, но сам по себе не доказывает освобождение CPU file mappings/weights. Host shard пока служит partition validation, а не model input production server.
- Production distributed atomic checkpoints/reconnect для native pipeline, journal integration в coding agent и восстановление всех KV/GDN/RNG owners ещё открыты.
- Distributed speculation/recurrent replay, weighted KV ownership, автоматическая thermal migration и совместное tail-compute + remote KV в этом launcher не включены. Эти комбинации требуют отдельного correctness gate.
- OpenCode 1.19.0 прошёл software config/SSE/reasoning/read/edit/test gate с fake backend. Другую установленную версию следует проверить отдельно; целевой Mac, настоящая модель, offline dependency installation и длительный coding loop остаются аппаратной приёмкой.

Следующая аппаратная процедура: [HARDWARE_RUNBOOK.md](HARDWARE_RUNBOOK.md). Измерения сохраняются отдельно от software receipts и исторических upstream результатов.
