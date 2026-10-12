# Приёмка на Mac, iPad и iPhone

Сначала выполнить [программные gates](SOFTWARE_ACCEPTANCE.md). Цель аппаратной сессии — проверить установленную сборку, locked model, обе пары и трёхстадийный pipeline; затем выбрать практически полезный режим по качеству, времени и памяти. Ни один software receipt не является результатом этого опыта.

## Подготовка

- Mac от питания, два разблокированных mobile workers в foreground, два data-кабеля и отдельные порты. Записать реально согласованный USB/Thunderbolt mode; надпись на кабеле не является замером.
- Заранее доступны weights, manifests, build toolchain, Python/npm/provider dependencies и local coding fixture. Не добавлять weights, endpoints, UDID, Team ID и системные логи в tracked files.
- Приложение BackCluster имеет действующую подпись и запускается standalone без debugger. Проверить срок sideload-подписи. Screen lock/background GPU проверять как отдельный failure scenario.
- Достаточно SSD для full GGUF, всех shards и checkpoints. Начать с одной квантизации и speculation off.

Все пути ниже локальные. Переменные `BC_MODEL`, `BC_LOCK`, `BC_REGISTRY`, `BC_PROFILE`, `BC_HOST_BUDGET`, `BC_HOST_RESERVE`, `BC_HOST_STATE`, `BC_IPAD_STATE`, `BC_IPHONE_STATE` заполняются фактическими файлами/замерами; оценки state включают KV, recurrent, graph и transfer buffers, а не только веса.

## 1. Установить одинаковый runtime

```bash
git submodule update --init --recursive
git rev-parse HEAD
git -C llama.cpp rev-parse HEAD
scripts/check.sh
scripts/test-native-pipeline.sh
./backcluster runtime-selftest --output .backcluster/runtime-selftest.json
./backcluster build
IPA=1 JOBS=3 scripts/build-backcluster.sh
```

Использовать installer BackCluster по [PRE_HARDWARE.md](PRE_HARDWARE.md); исходный upstream installer не запускается как обновление BackCluster. Engine base commit и content-addressed overlay build ID должны совпадать на Mac и обоих workers. Не делать `submodule update --remote` между baseline и candidate.

## 2. Проверить endpoints, profile и links

Сначала подключить каждый worker отдельно и определить его cable interface/endpoint. Затем заполнить private registry по `profiles/registry.example.json`: distinct routes, expected hardware model, measured allocation budget/reserve, актуальные protocol/engine/build versions. Registry pseudonyms не являются криптографической hardware authentication.

```bash
./backcluster doctor --registry "$BC_REGISTRY" --human
./backcluster profile --registry "$BC_REGISTRY" --output "$BC_PROFILE"
./backcluster link-bench --registry "$BC_REGISTRY" --repeats 5 --output .backcluster/link-serial.json
./backcluster link-bench --registry "$BC_REGISTRY" --repeats 5 --concurrent --output .backcluster/link-concurrent.json
```

Сохранить physical/available/footprint/Metal bytes и thermal state. Не заменять неизвестный лимит числом по physical RAM. Payload bench включает разные размеры, upload ACK, download и raw samples; это application bandwidth, а не паспортная полоса. Не проводить его одновременно с inference. При отказе route/identity/admission исправить исходную причину и повторить probe.

## 3. Закрепить реальную модель и shards

`model --file ... --repository ... --revision ... --output ...` создаёт full source lock; `model --file ... --check ...` сверяет bytes и metadata. Revision — immutable 40-character SHA. Файл с похожим именем и незавершённая загрузка не проходят content verification.

Выбрать границы `BC_L` и `BC_M` по trunk/full-attention metadata и измеренным budgets. Для пары нужен tail `[L,N)`. Для pipeline нужны host `[0,L)`, middle `[L,M)`, tail `[M,N)`; MTP не включать в trunk по предположению. Использовать `shard --help` и создать три manifests, затем:

```bash
./backcluster shard-check --file .backcluster/host.gguf --manifest .backcluster/host-shard.json --source "$BC_MODEL"
./backcluster shard-check --file .backcluster/middle.gguf --manifest .backcluster/middle-shard.json --source "$BC_MODEL"
./backcluster shard-check --file .backcluster/tail.gguf --manifest .backcluster/tail-shard.json --source "$BC_MODEL"
./backcluster partition-check --manifest .backcluster/host-shard.json --manifest .backcluster/middle-shard.json --manifest .backcluster/tail-shard.json
```

Для middle используется role `middle` без output head; tail для decode сохраняет output head. Копировать соответствующий mobile shard в `Documents/tail.gguf` приложения каждого worker, затем перезапустить его. Имя файла одинаковое; roles/ranges различаются по GGUF metadata. После eager load новый profile должен содержать `loaded_tail` с точным hash/source/range/head/build identity. `null` означает, что strict compute launch ещё не разрешён.

## 4. Mac-only baseline

```bash
./backcluster serve --file "$BC_MODEL" --lock "$BC_LOCK" --context 16384 --dry-run --output .backcluster/mac-launch.json
./backcluster serve --file "$BC_MODEL" --lock "$BC_LOCK" --context 16384 --output .backcluster/mac-launch.json
```

В другом terminal проверить `/health`, `/v1/models`, затем:

```bash
./backcluster quality --lock "$BC_LOCK" --launch .backcluster/mac-launch.json --output .backcluster/mac-quality.json
./backcluster bench --lock "$BC_LOCK" --launch .backcluster/mac-launch.json --prompt .backcluster/prompt.txt --repeats 5 --tokens 256 --output .backcluster/mac-bench.json
```

Не повышать context до 32k/64k до измерения пика и reserve с открытым IDE. Не менять глобальный wired limit ради прохождения теста. Baseline без draft/speculation сохраняет тот же cache dtype и prompt, что candidate.

## 5. Пары и pipeline

Сначала `split-decode`/`capacity` с одним worker role `tail_stage`, затем другая пара с новым exact tail range и profile. `split-prefill` — отдельная серия с полной моделью на Mac; она не доказывает capacity gain. Для pipeline registry содержит iPad `middle_stage` и iPhone `tail_stage` либо роли, выбранные по замерам.

Пример полной pipeline command; до фактического запуска проверить тот же набор параметров с `--dry-run`:

```bash
./backcluster serve --mode pipeline --file "$BC_MODEL" --lock "$BC_LOCK" \
  --registry "$BC_REGISTRY" --profile "$BC_PROFILE" --context 16384 \
  --host-budget-bytes "$BC_HOST_BUDGET" --host-reserve-bytes "$BC_HOST_RESERVE" \
  --host-state-bytes "$BC_HOST_STATE" \
  --state-bytes "ipad=$BC_IPAD_STATE" --state-bytes "iphone=$BC_IPHONE_STATE" \
  --split-layer "$BC_L" --middle-end "$BC_M" \
  --middle-file .backcluster/middle.gguf --middle-manifest .backcluster/middle-shard.json \
  --tail-file .backcluster/tail.gguf --tail-manifest .backcluster/tail-shard.json \
  --output .backcluster/pipeline-launch.json
```

Live checks выполняются повторно, даже если передан profile. Сначала качество/greedy correctness, затем пять одинаковых bench repetitions. Сравнить:

```bash
./backcluster quality --lock "$BC_LOCK" --launch .backcluster/pipeline-launch.json --output .backcluster/pipeline-quality.json
./backcluster bench --lock "$BC_LOCK" --launch .backcluster/pipeline-launch.json --prompt .backcluster/prompt.txt --repeats 5 --tokens 256 --output .backcluster/pipeline-bench.json
./backcluster bench-compare --baseline .backcluster/mac-bench.json --candidate .backcluster/pipeline-bench.json --allow-runtime-changes --output .backcluster/comparison.json
```

`--allow-runtime-changes` разрешает явное сравнение разных режимов; source/prompt/output evidence всё равно проверяются. Не выдавать different tokenization/context/cache/quantization за чистое distributed сравнение. Проверить короткий decode, cold prefill, cached append и reset отдельно. При различии greedy output сравнить teacher-forced logits/top-k на той же модели прежде, чем делать вывод о качестве.

`remote-kv` с одним/двумя workers — отдельный режим и отдельный registry role. Его total context больше local window, соответствующие state estimates учитывают block-cyclic allocation; tail-compute и remote KV одновременно launcher отвергает.

## 6. Отказы, память и offline coding

Провести час нагрузки с IDE и локальной сборкой: peak app footprint, Metal allocation, Mac resident memory/swap, thermal, wall time и accepted tok/s. Capacity требует фактического уменьшения Mac peak memory; tensor override и сумма RAM трёх устройств этого не доказывают.

Отменить SSE, отключить каждый кабель во время prefill/decode, свернуть/заблокировать worker и подключить снова. После потерянного уникального состояния ожидается bounded failure; продолжение с неизвестным KV/GDN запрещено. Для текущего native pipeline безопасный recovery — clear/re-prefill полной согласованной истории. Production distributed checkpoint recovery пока не принят; BCTX checkpoint test не использовать как его substitute.

Запустить `tests/security/run.sh` на Mac и exposure check по справке `scripts/check-phone-exposure.py`. Проверить raw-port отказ на Wi-Fi и действующую paired encrypted tunnel policy.

Определить `opencode --version`, затем `opencode-config --client-version ... --lock ... --context ... --output ...`; provider URL/alias сверить с `/v1/models`. Выполнить реальную read/edit/test задачу, tool round-trip, reasoning/content parsing и cancel. Последнюю серию провести без внешних сетей, сохранив кабели; никакие packages/weights/remote LLM не должны догружаться.

Сохранить private raw reports и обезличенный итог: commits/build IDs, hashes, установки/signing, routes/link modes, context/cache settings, memory/thermal peaks, качество и сбои. Публиковать только проверенные результаты; пути, имена, endpoints, UDID и credentials удалить.
