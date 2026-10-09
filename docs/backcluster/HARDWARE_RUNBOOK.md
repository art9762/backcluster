# Первый заход на железо

Цель первой сессии: воспроизвести baseline и сравнить две пары. Трёхстадийный runtime пока не реализован. Не пытаться получить его комбинацией существующих env flags.

## До запуска

- Mac, iPad и iPhone физически рядом; Mac от питания, оба мобильных устройства разблокированы.
- Два data-кабеля и отдельные порты Mac. Для iPhone нужен кабель USB 3 на 10 Gb/s; charging cable может оказаться USB 2. Для iPad проверяем реально согласованный режим, даже с Thunderbolt-кабелем.
- Достаточно свободного SSD: основная модель, drafter, отдельные shards и checkpoints занимают больше одного GGUF. Посчитать по выбранным файлам; не скачивать все квантизации сразу.
- Xcode/CLI tools, CMake и Python dependencies установлены заранее. Подпись приложения проверена на обоих устройствах. Срок действия sideload-подписи учитывается перед поездкой без сети.
- Версии ОС, чипов, core counts и тип установки записаны. Проверяется standalone app без debugger.
- Native worker остаётся foreground. Guided Access при необходимости; экран не должен блокироваться в середине опыта.

## 1. Проверить source identity

В корне клона BackCluster:

```bash
git submodule update --init --recursive
git rev-parse HEAD
git -C llama.cpp rev-parse HEAD
python3 bench/device-probe.py --no-phone
```

Ожидаемый engine pin в этой подготовке: `9f17fc31cf60753eb6bd6ec075894edab773cb91`. Не обновлять его автоматически до baseline. Для mobile build и установки следовать закреплённому [upstream README](../../README.upstream.md) и [install guide](../INSTALL-IPHONE.md); будущий установщик BackCluster ещё не готов.

Приложение и Mac должны использовать совместимую engine/state version. Team ID и UDID вводятся только локально; не записывать их в tracked files.

## 2. Подключать workers по одному

Открыть установленное приложение, сначала только iPhone, затем только iPad:

```bash
python3 bench/device-probe.py
```

Probe может попытаться перезапустить app через `phone-up.sh`. Сохранять только обезличенную выдачу. Записать физическую RAM, app available bytes, footprint, wired/system reading и thermal; неизвестные значения не заменять нулями. Затем подключить оба и убедиться, что registry/probe различает устройства и интерфейсы.

Отдельный BC-002 benchmark должен измерять RTT и throughput на 10–20 KB, ~1 MB и ~2.6 MB payload в обе стороны, а также два одновременных потока. Существующий probe измеряет маленький round-trip; этого недостаточно для заявления о полосе.

## 3. Закрепить модель и собрать engine

Использовать snapshot из [UPSTREAM.md](../../UPSTREAM.md), записать фактические bytes/SHA-256 и metadata. Не считать скачивание завершённым только по наличию непустого файла. Начинаем с IQ4_XS, проверяем модель без drafter, затем совместимый DFlash2. Не скачивать weights в git.

```bash
cmake -S llama.cpp -B llama.cpp/build-metal -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-metal --target llama-server llama-quantize -j
```

Первый консервативный пример использует существующий upstream launcher. Он требует, чтобы IQ4-файл лежал по его documented default path; иначе передать корректный `MODEL` локально:

```bash
PHONE=0 PHONE_ANE=0 DRAFT= SPEC_TYPE=none SPEC_REPLAY=0 \
SME=0 MM_SME=0 PROXY=0 LOAD_MODE=mmap CTX=16384 \
scripts/serve.sh
```

Это диагностический профиль, не оптимальный режим: mmap может давать page-ins. Проверить `/health`, `/v1/models`, короткий ответ и memory pressure. Затем отдельно исследовать upstream wired mode с фактическим запасом RAM. Не применять глобальный `sysctl` вслепую; записать исходное значение и выбранный лимит, проверить параллельную работу IDE.

## 4. Baseline и пары

| Серия | Состав | Что отделяет |
|---|---|---|
| B0 | Stock llama.cpp, если поддерживает данный GGUF | Эффект fork kernels; необязателен для pair-сравнения |
| B1 | Pinned fork, Mac, draft off | Базовая математика и память |
| B2 | Pinned fork, Mac, draft on | Эффект speculative decoding |
| B3 | Mac + iPhone | Эффект одного worker |
| B4 | Mac + iPad | Выбор лучшей пары |
| B5 | Mac + два KV-workers | Проверка remote context; отдельная глубокая серия |
| B6 | Distributed weights, 2/3 стадии | Только после соответствующей реализации |

Порядок B2/B3/B4 чередовать, учитывать одинаковый thermal state и cache state. 16k/32k сначала, затем 64k при наличии бюджета. Prefill: короткие ~128-token turns, append ~2k и cold ~8k; decode: 256 accepted output tokens. Пять повторов, raw rows и median/p95. Первый restore и дальнейшие appends не смешивать.

После проверки безопасного 64k upstream profile можно переиспользовать:

```bash
python3 bench/turn-bench.py --build
python3 bench/turn-bench.py --config mac --reps 5
python3 bench/turn-bench.py --config phone --reps 5
```

`turn-bench.py` жёстко задаёт 64k, запускает server и пишет в `bench/results/turns.jsonl`. Для нового режима сначала адаптировать параметры/outputs, не считать его маленьким smoke-test. Не перезаписывать и не выдавать historical upstream rows за новые. Выводы BackCluster хранить отдельно с source commit и timestamp.

Для выбора iPad или iPhone в раннем парном тесте физически подключать только нужный worker. С несколькими устройствами порядок первой строки auto-discovery не является выбором роли.

## 5. Качество, устойчивость, agent loop

- Greedy regression на фиксированных prompts и context sizes; при расхождении сравнить teacher-forced top-1/top-k и KL. Квантизацию не менять внутри distributed-vs-local проверки.
- `launch-quality.py` для probabilities запускать без speculation: upstream сообщает ограниченное покрытие probabilities при draft mode.
- 60 минут нагрузки: peak footprint, Metal memory, swap delta, temperature/thermal state, power и accepted tok/s. Проверить с открытым IDE и локальной сборкой.
- Контролируемо отменить streaming request; свернуть приложение; заблокировать worker; отключить кабель; подключить снова. Ожидаемые outcomes описаны в [архитектуре](ARCHITECTURE.md).
- Выполнить `tests/security/run.sh` на Mac и `scripts/check-phone-exposure.py` по его справке на устройствах после сетевых изменений. В Linux-подготовке suite не запускался из-за отсутствия Apple toolchain.

На OpenCode: определить `opencode --version`, выбрать подходящий provider schema, использовать `http://127.0.0.1:8080/v1` и model ID, проверенный через `/v1/models`. Ограничение context клиента должно соответствовать серверу и оставлять output budget. Проверить thinking/reasoning отдельно от финального content и корректный round-trip tool calls.

## 6. Offline-приёмка

Все weights, draft, Python/npm dependencies, provider package, build toolchain и локальные тестовые зависимости предварительно доступны. Выключить внешние сети, сохранив кабельный transport. Запустить новую coding-сессию и завершить read/edit/test задачу без удалённого LLM, загрузки packages или web-tools. IDE/agent telemetry и auto-update проверить по фактической версии клиента.

## Что принести в следующую сессию

Обезличенный device profile, versions и artifact hashes; B1–B4 raw results; installed signing type; фактические link modes; симптомы memory/thermal failures. Сырые системные логи перед публикацией проверить на пути, имена, UDID и credentials.
