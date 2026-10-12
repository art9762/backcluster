# Подготовка к аппаратной приёмке, BackCluster 0.4

Есть Mac/pair/remote-KV/pipeline launchers, streaming shards, проверки качества и
сравнения. Native CPU gate проверяет настоящий трёхстадийный forward на маленькой
детерминированной Qwen35 с GDN и full attention. Это не проверка качества
Qwen3.8-27B или Apple kernels. `hardware_validated=false` и offline
`plan.launch_ready=false` сохраняются. Подробная матрица:
[SOFTWARE_ACCEPTANCE.md](SOFTWARE_ACCEPTANCE.md).

## 1. Собрать один runtime для Mac и workers

```bash
git submodule update --init --recursive
scripts/check.sh
scripts/test-native-pipeline.sh
# Apple Silicon Mac, Python 3.9+, CMake, Xcode 26+ и Metal toolchain:
./backcluster doctor --human
./backcluster build
IPA=1 scripts/build-backcluster.sh
```

Engine gitlink сохранён. `engine-patches/manifest.json` задаёт точный base и hashes
downstream patches. Сборка проверяет base/patches и материализует source в ignored
`.backcluster/engine-source/`, не изменяя submodule. Cache сравнивается с
независимой реконструкцией. Host/app публикуют одинаковый build ID. Изменённые
patches, source или server/runtime binaries отвергаются. Если CMake старого
`llama.cpp/build-metal` указывает другой source directory, убрать эту generated
сборку перед первым build 0.4.

Native gate работает на Linux, генерирует untrained веса без скачивания и
проверяет f32/f16 residual, logits/greedy, append/reset, framing/identity/epoch и
потерю обоих workers. Для software-only toolkit также:
`./backcluster runtime-selftest`.

IPA: `ios/build/BackCluster.ipa`. Для собственной подписи/установки:

```bash
DEVELOPMENT_TEAM=<local-team-id> UDID=<local-device-id> scripts/build-backcluster.sh
```

Ad-hoc IPA для AltStore не доказывает entitlement или standalone budget.
Workers держать foreground, без debugger. Manual build без generated identity
header не проходит cluster compatibility. App bundle: `app.backcluster.sidecar`;
device helpers поддерживают AltStore suffix и требуют явный выбор при ambiguity.

## 2. Снять profiles и задать роли

```bash
./backcluster profile --worker "ipad=${IPAD_CABLE_IP}" --worker "iphone=${IPHONE_CABLE_IP}" --output .backcluster/trio.json
./backcluster link-bench --worker "ipad=${IPAD_CABLE_IP}" --worker "iphone=${IPHONE_CABLE_IP}" --concurrent --output .backcluster/links.json
```

Скопировать `profiles/registry.example.json` в `.backcluster/registry.json`.
Записать реальные cable IPv4, разные Mac interfaces, model identifiers,
protocol versions и `engine_build_id` из worker profile; budgets/reserves —
измеренные integer bytes. Null в примере намеренно запрещает launch.

Роли: `remote_kv`, `tail_stage`, `middle_stage`, `idle`. Для пары нужны один tail
и idle; для pipeline middle и tail; для remote KV один/два KV workers.
Для экспериментов хранить отдельные registry-файлы. Порядок remote-KV workers
определяет block-cyclic ownership. Default ports: control 50061, tail 50060,
KV 50062.

```bash
./backcluster profile --registry .backcluster/registry.json --output .backcluster/trio.json
./backcluster doctor --registry .backcluster/registry.json --human
```

Route проверяется до probe. Снимок старше пяти минут блокирует launch spec;
live `serve` повторяет profile/identity/thermal/admission перед exec. Worker
budget — дополнительный allocation limit; уже verified resident weights
повторно не учитываются. `--host-state-bytes` и `--state-bytes alias=bytes`
задают peak KV/recurrent/compute/transfer state отдельно от weights. Это
операторские оценки, которые надо измерить. RAM устройств не суммируется.

Link bench измеряет 16 KiB/1 MiB/2600 KiB, warmup и raw repeats. Payload integrity
обязательна. Effective bandwidth включает application transport; не объявляется
физической полосой USB. Не запускать одновременно с inference.

## 3. Закрепить source и shards

```bash
./backcluster model --file "$MODEL" --repository "$MODEL_REPOSITORY" --revision "$MODEL_REVISION" --output .backcluster/model-lock.json
./backcluster model --file "$MODEL" --check .backcluster/model-lock.json
```

Нужны immutable 40-hex publisher revision и полный unsplit GGUF v3. Проверяются
SHA-256/bytes, engine base, dtype/shape, tokenizer/template, tensor bytes слоёв,
output/embedding/MTP. Lock не доказывает заявленное происхождение файла.
Чтение/hashing выполняются потоково; multipart сначала объединить.

Выбрать `L1`, `L2`, `TRUNK_LAYERS` по metadata и budgets. Границы кратны
full-attention interval. Для трёх стадий:

```bash
./backcluster shard --file "$MODEL" --lock .backcluster/model-lock.json --start 0 --end "$L1" --role host --output .backcluster/host.gguf --manifest .backcluster/host-shard.json
./backcluster shard --file "$MODEL" --lock .backcluster/model-lock.json --start "$L1" --end "$L2" --role middle --output .backcluster/middle.gguf --manifest .backcluster/middle-shard.json
./backcluster shard --file "$MODEL" --lock .backcluster/model-lock.json --start "$L2" --end "$TRUNK_LAYERS" --role tail --output .backcluster/tail.gguf --manifest .backcluster/tail-shard.json
./backcluster partition-check --manifest .backcluster/host-shard.json --manifest .backcluster/middle-shard.json --manifest .backcluster/tail-shard.json
./backcluster shard-check --file .backcluster/middle.gguf --manifest .backcluster/middle-shard.json --source "$MODEL"
```

Manifest сохраняет source/tokenizer/template identity, global layer map и
transform. Проверка с `--source` сравнивает фактические bytes каждого tensor.
Middle без head, tail с head. MTP по умолчанию удалён; `--keep-mtp` только у tail.
`scripts/split-gguf.py` поддерживает диапазоны и старый `-L`, без numpy.

Передать middle/tail соответствующим устройствам и запустить tail listener,
как в [HARDWARE_RUNBOOK.md](HARDWARE_RUNBOOK.md). Worker хеширует файл до load,
проверяет его неизменность и публикует `loaded_tail` без пути/имени. Изменение
файла после load инвалидирует receipt. Launcher сверяет оба фактических hashes.

## 4. Baseline, пары и pipeline

```bash
./backcluster serve --lock .backcluster/model-lock.json --file "$MODEL" --context 16384 --output .backcluster/mac-launch.json
# Во втором terminal:
./backcluster quality --lock .backcluster/model-lock.json --launch .backcluster/mac-launch.json --tokens 512 --output .backcluster/mac-quality.json
./backcluster bench --lock .backcluster/model-lock.json --launch .backcluster/mac-launch.json --prompt bench/prompt.txt --repeats 5 --tokens 256 --output .backcluster/mac-bench.json
```

Endpoint `http://127.0.0.1:8080/v1`, alias `backcluster-<16 hex model SHA-256>`.
Одна сессия, q8 KV, drafter/speculation/SME off, loopback, без sysctl/discovery.
`--dry-run` выдаёт private receipt без exec.

Для пары создать tail `[L1,TRUNK_LAYERS)` и registry с одним `tail_stage`.
`--mode split-prefill` хранит полную модель на Mac и offloads большие prefills;
`--mode capacity` (alias `split-decode`) отправляет каждый ubatch на tail.
Задать snapshot, host budget/reserve/state, worker state, `--split-layer`,
`--tail-file`, `--tail-manifest`.

Полный pipeline:

```bash
./backcluster serve --mode pipeline --lock .backcluster/model-lock.json --file "$MODEL" \
  --registry .backcluster/pipeline-registry.json --profile .backcluster/trio.json \
  --host-budget-bytes "$HOST_BUDGET" --host-reserve-bytes "$HOST_RESERVE" --host-state-bytes "$HOST_STATE" \
  --state-bytes "ipad=$IPAD_STATE" --state-bytes "iphone=$IPHONE_STATE" \
  --split-layer "$L1" --middle-end "$L2" \
  --middle-file .backcluster/middle.gguf --middle-manifest .backcluster/middle-shard.json \
  --tail-file .backcluster/tail.gguf --tail-manifest .backcluster/tail-shard.json \
  --context 16384 --output .backcluster/pipeline-launch.json
```

Host использует полный GGUF на диске; удалённые tensors остаются CPU/mmap,
локальный decode graph их не выполняет. Host derived shard служит verification,
а не входом launcher. Реальное снижение resident RAM и page-ins измерить на Mac.
Контекст compute-split ограничен launcher до 65536; context shifting отключён.
Потеря unique stage завершает запрос, продолжение требует reset/re-prefill0.

`--mode remote-kv` требует один/два KV workers и `--total-context` больше локального
`--context`. Проверяются head geometry, точное число block-cyclic positions и
минимальные KV bytes каждого worker. Compute-split+remote KV отвергается.

`plan` перебирает capacity boundaries по реальным tensor bytes; требует
`--host-state-bytes` для feasible estimate. Pipeline задаётся `--split-layer`
и `--middle-end`. Он остаётся оценкой, не разрешением пропустить live checks.
Errors: 1; argparse: 2; Ctrl-C до exec: 130; после exec exit code сохраняет server.

## 5. Сравнение и OpenCode

После того же quality gate получить candidate benchmark с его launch receipt:

```bash
./backcluster bench-compare --baseline .backcluster/mac-bench.json --candidate .backcluster/pipeline-bench.json --output .backcluster/comparison.json
./backcluster opencode-config --lock .backcluster/model-lock.json --context 16384 --client-version "$OPENCODE_VERSION" --output .backcluster/opencode.json
OPENCODE_CONFIG="$PWD/.backcluster/opencode.json" opencode
```

Model/build/prompt/context/sampling должны совпадать. Raw medians пересчитываются;
при output/token mismatch speed ratio не выдаётся. Placement/batch/split
differences перечислены; tuning внутри одного режима требует
`--allow-runtime-changes`. Usage берётся из API, SSE chunks не считаются токенами.
TTFT/wall accepted tok/s не является pure decode rate. Reports не содержат prompt
или output text; model alias не доказывает loaded weights другого процесса.

`quality` проверяет JSON/SSE, отдельное reasoning и фиксированный read/edit/test
цикл в tempdir. Закрытие stream не доказывает остановку GPU. OpenCode config
поддерживает явную stable 1.x; другие схемы требуют проверки. Настоящий
OpenCode 1.19.0 принял config в software check; это не аппаратный coding loop.
`--reasoning` включать после наблюдения отдельного reasoning output.

## Границы

Осталась аппаратная приёмка реализованных режимов: реальная модель, Apple
kernels, memory/thermal/link, signing, OpenCode и длительная нагрузка.
Weighted KV, compute-split+remote KV и distributed speculation ещё не реализованы.
BCTX durable checkpoints/journal — проверенный отдельный toolkit; production
native OpenAI checkpoint/sampler/transcript integration остаётся программной
работой. В launcher native slot cache/checkpoints выключены.

Локальные weights/registry/reports хранить в ignored `.backcluster/`.
Launch receipts содержат пути/endpoints: проверить перед публикацией.
Полная последовательность: [HARDWARE_RUNBOOK.md](HARDWARE_RUNBOOK.md).
