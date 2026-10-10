# Программная подготовка и следующая сессия на устройствах

CLI 0.3.0 подготовлен для первого аппаратного этапа. Автоматические тесты проверяют форматы, отказные сценарии и transport, но не подтверждают работу модели на целевой связке. `hardware_validated=false`, `cluster_ready=false` и `plan.launch_ready=false` сохраняются намеренно.

## Что реализовано

| Команда | Что проверяет |
|---|---|
| `doctor`, `profile` | Host, worker, версии, наблюдения памяти; с registry также маршрут и совместимость |
| `model` | SHA-256/bytes, immutable source revision, pin engine, GGUF v3, template hash, dtype layout, bytes каждого слоя |
| `plan` | Явные роли, compatibility, отдельные memory estimates; печатает блокеры без запуска |
| `link-bench` | 16 KiB, 1 MiB, 2600 KiB, целостность payload, upload+ack/download, пять повторов и warmup |
| `build` | Собирает закреплённый Mac `llama-server` без автоматического обновления |
| `serve` | Проверяет весь GGUF перед запуском Mac-only; loopback, draft off, mmap, q8 KV, одна сессия |
| `bench` | Локальный streaming baseline, TTFT/wall time, usage token count, private raw rows и median/p95 |

`model` записывает заявленное происхождение локального файла и проверяет его содержимое. Он не скачивает веса и не доказывает, что файл получен от указанного publisher. Изменение файла, metadata, bytes или engine pin делает `--check` и `serve` ошибкой. Парсер не поддерживает multipart GGUF и неизвестные tensor types; их нельзя молча пропускать. Размеры tensors берутся из их фактических shapes и storage types.

Derived shards и drafter compatibility ещё требуют отдельного manifest и проверки на закреплённом engine после BC-003. Карта слоёв не заменяет проверку output tensors, MTP и численной корректности shard. `serve` отклоняет split artifacts.

## 1. Получить рабочую ветку и собрать приложение

```bash
git fetch origin
git switch main
git pull --ff-only
git submodule update --init --recursive
./backcluster --version
./backcluster doctor --human
./backcluster build
IPA=1 scripts/build-backcluster.sh
```

Нужны Apple Silicon, Python 3.9+, CMake, Xcode 26+ и Metal toolchain. `doctor` до установки workers может быть неуспешным: это диагностический результат, а не причина считать проверку кластера пройденной.

CI собирает complete device archive и выкладывает `BackCluster-device-build` с `BackCluster.ipa`. Этот IPA ad-hoc signed для последующей установки через AltStore, не подписан Apple developer team. CI не подтверждает signing entitlement, app budget без debugger или запуск на твоих устройствах. Для своей подписи:

```bash
DEVELOPMENT_TEAM=<локальный-team-id> UDID=<локальный-device-id> scripts/build-backcluster.sh
```

По умолчанию новый bundle ID `app.backcluster.sidecar`, display name BackCluster. Для отдельных provisioning profiles можно переопределить `APP_BUNDLE_ID` локально. Installer upstream не вызывается. Helper `bundle-id.sh` ищет установленный BackCluster (включая AltStore suffix), затем Backburner; несколько совпадений требуют явного `BUNDLE_ID`. Он не выводит предполагаемый upstream bundle ID только по `DEVELOPMENT_TEAM`. Приложение продолжает использовать upstream storage/tunnel semantics; поведение полного runtime проверяем после установки. При явной установке на устройство скрипт возвращает ошибку, если `devicectl install` не удался.

В worker profile добавлены `engine_commit` и `control_protocol_version=1`; commit берётся из дерева, из которого собран framework. Скрипт запрещает другой/dirty engine. Ручная сборка без generated build header сообщает `engine_commit=unknown` и не проходит registry compatibility. Legacy profile остаётся читаемым для diagnostics, но не проходит registry admission.

## 2. Первые profiles и два link tests

Открыть приложение foreground, без debugger. Сначала подключить только одно устройство и выписать его cable IP из UI. Второе подключить после отдельной проверки. Перед командами задать локально `IPAD_CABLE_IP` и `IPHONE_CABLE_IP` по UI приложения:

```bash
./backcluster profile --worker "ipad=${IPAD_CABLE_IP}" --output .backcluster/ipad.json
./backcluster profile --worker "iphone=${IPHONE_CABLE_IP}" --output .backcluster/iphone.json
./backcluster profile --worker "ipad=${IPAD_CABLE_IP}" --worker "iphone=${IPHONE_CABLE_IP}" --output .backcluster/trio.json
./backcluster link-bench --worker "ipad=${IPAD_CABLE_IP}" --output .backcluster/link-ipad.json
./backcluster link-bench --worker "iphone=${IPHONE_CABLE_IP}" --output .backcluster/link-iphone.json
./backcluster link-bench --worker "ipad=${IPAD_CABLE_IP}" --worker "iphone=${IPHONE_CABLE_IP}" --concurrent --output .backcluster/link-trio.json
```

Одна операция использует отдельное control connection; worker выделяет максимум 4 MiB и завершает передачу по общему deadline 10 s. Неверная версия, payload mismatch, EOF и timeout дают ошибку и сохраняются в отчёте. Тест не загружает модель, не меняет KV, не создаёт файл на worker и не добавляет listener. Сетевые cable/tunnel gates сохраняются.

`upload_ack_ns` включает отправку и получение подтверждения полного payload; `download_ns` начинается после ack и заканчивается на последнем байте echo. Это effective application bandwidth, не физическая полоса USB. `round_trip_ns` измеряет полный payload round trip без connect/ready handshake. Link reports не доказывают прямой cable transport (`transport=unverified`) и не должны запускаться одновременно с inference. Два потока параллельны по workers, не являются duplex benchmark одного link.

## 3. Явный registry после наблюдений

Скопировать `profiles/registry.example.json` в `.backcluster/registry.json`. Пример намеренно не проходит validation: budgets/reserves равны null. Заменить IP, Mac interfaces, model identifiers и добавить измеренные integer bytes. Поле `allocation_budget_bytes` — предел **дополнительных** allocations в данной сессии; `reserve_bytes` вычитается из наименьшего доступного лимита. Не брать эти значения из 12 ГБ физической RAM.

Registry проверяет разные cable interfaces и endpoints, `expected_model`, engine SHA и все protocol versions. Проверка live route выполняется до probe (`route -n get` должен показать заданный interface). Это local binding к классу устройства, не криптографическая hardware identity; два одинаковых iPad нельзя отличать только по `expected_model`. В этой связке iPad и iPhone имеют разные модели. Registry не выбирает первую строку discovery и не меняет маршруты.

```bash
./backcluster profile --registry .backcluster/registry.json --output .backcluster/trio.json
./backcluster doctor --registry .backcluster/registry.json
./backcluster link-bench --registry .backcluster/registry.json --concurrent --output .backcluster/link-trio.json
```

`plan` берёт минимум из explicit additional budget, process available, physical minus footprint и Metal recommended minus allocated, затем вычитает reserve. Неизвестное наблюдение, недостаточный бюджет или serious/critical/unknown thermal отклоняет estimate. Physical headroom не учитывает всех потребителей системы, поэтому этот расчёт не гарантирует отсутствие jetsam. Runtime/KV state needs измеряются отдельно. Снимок старше пяти минут добавляет blocker; cached snapshot никогда не разрешает live launch.

## 4. Зафиксировать локально скачанный GGUF

Основную модель выбрать и скачать по immutable revision из `UPSTREAM.md`. Убедиться, что это один полный GGUF поддерживаемой pinned engine архитектуры. Значения в команде ниже — локальные placeholders, не готовый lock:

```bash
./backcluster model --file /path/model.gguf --repository owner/model-repository --revision <40-hex-revision> --output .backcluster/model-lock.json
./backcluster model --file /path/model.gguf --check .backcluster/model-lock.json
```

Путь не сохраняется в model lock; имя файла сохраняется. GGUF tensors не читаются в RAM целиком. Проверяются alignment, tensor spans/overlap, все layer indexes, metadata limits, shape и storage size известного типа. Чтение/hashing большого файла может занять несколько минут. Hash доказывает воспроизводимость, не поддержку модели.

## 5. Mac baseline и review plan

```bash
./backcluster serve --lock .backcluster/model-lock.json --file /path/model.gguf --context 16384 --dry-run
./backcluster serve --lock .backcluster/model-lock.json --file /path/model.gguf --context 16384 --output .backcluster/launch.json
```

CLI запускает бинарник напрямую, очищает inherited inference/proxy/loader overrides, выключает SME helpers, speculation и remote workers. Он не изменяет global sysctl, не выбирает worker автоматически и не скачивает dependencies. `--dry-run` проверяет identity файла и показывает argv, но не подтверждает сборку/память. `serve` требует Apple Silicon, clean engine pin и ранее собранный binary. mmap baseline может давать page-ins; после первого замера исследовать wired и SME отдельно.

Во втором terminal, с фиксированным prompt и тем же запущенным server:

```bash
./backcluster bench --lock .backcluster/model-lock.json --prompt bench/prompt.txt --repeats 5 --tokens 256 --output .backcluster/baseline-16k.json
```

Server alias: `backcluster-<первые 16 hex SHA256>`. Bench запрашивает именно его, но сохраняет `server_identity_verified=false`: OpenAI API сам по себе не доказывает hash загруженных весов или engine. Для сравнения необходимо приложить launch receipt, lock и source revision. Повторить с `serve --context 32768`, только если 16k проходит бюджет. Requests sequential, один warmup отдельно от пяти measured repeats. В report нет prompt/output text; есть их hashes. TTFT включает reasoning и tool-call output. `output_hash_format=normalized_choices_v1` учитывает текст, reasoning и функции инструментов независимо от разбиения SSE; случайные tool-call IDs исключены. Эти hashes не сравниваются с raw-text hashes старого runner. Token count берётся из SSE usage; при его отсутствии остаётся null, chunks не объявляются токенами. Accepted tokens/s — токены на полный wall time, включая prefill; это не pure decode speed. Ошибка stream/deadline сохраняет неуспешную строку. Server может не сразу заметить закрытие сокета; фактическая cancellation остаётся аппаратной проверкой BC-006.

Review plan на свежем snapshot требует явных host budget/reserve в байтах:

```bash
./backcluster plan --lock .backcluster/model-lock.json --profile .backcluster/trio.json --registry .backcluster/registry.json --mode remote-kv --host-budget-bytes <measured-budget> --host-reserve-bytes <reserve> --state-bytes ipad=<measured-state> --state-bytes iphone=<measured-state> --output .backcluster/plan.json
```

Одна assignment у Mac включает все веса; estimates не изображают реализованный capacity scheduler. `plan` exit 0 означает корректный отчёт, а не разрешение запуска. Model/profile/link/serve/bench errors дают 1; argparse errors дают 2. Неуспешный `build` даёт 1 с кодом неуспешной команды в сообщении; Ctrl-C до exec даёт 130. После exec сервер сохраняет свой exit code.

## Следующая обязательная граница

Дальнейшие этапы требуют devices: установить app и проверить entitlement/budget standalone, собрать live profiles/link samples, запустить один forward, закрепить фактический model artifact и сравнить Mac-only с каждой парой. Затем выбрать роли и split по latency/memory. До этого BC-012/021/023/024 и полноценный OpenCode offline acceptance остаются открытыми. Трёхстадийного inference и объединённого пула 48 ГБ ещё нет.

Все локальные reports писать в ignored `.backcluster/`. Launch receipt содержит путь к весам, registry — адреса/interfaces; проверить их перед публикацией. Existing `bench/results/` — historical upstream data; новый runner его не меняет.

## Проверки разработки

`scripts/check.sh` запускает Python regression tests, shell syntax checks, listener lint, sensitive-data hook tests и native C++ payload tests. `doctor --human` выводит читаемые результаты с подсказками; `--output` при этом сохраняет JSON. Новый `install.sh` устанавливает CLI из текущего checkout. Старый downloader сохранён как `scripts/install-upstream.sh` и относится к upstream, а не к выпуску BackCluster.
