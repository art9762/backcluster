# P1: диагностика устройств

Реализован первый программный шаг BC-001: CLI и отдельный read-only `profile`
на существующем control port мобильного worker. Полная приёмка P1 требует
Mac/iPad/iPhone, standalone-приложения и короткого forward. Эти замеры ещё не
выполнены. `model`, `shard`, `plan`, `serve`, `bench` и `quality` дополняют эту
диагностику; актуальные команды и границы проверки описаны в
[PRE_HARDWARE.md](PRE_HARDWARE.md).

## Запуск

Python 3.9+; сторонние Python-пакеты и интернет для команд не нужны.

```bash
./backcluster --version
./backcluster doctor
./backcluster profile --output .backcluster/host-profile.json
./backcluster profile --check .backcluster/host-profile.json
```

`doctor` проверяет Apple Silicon host, физическую RAM, checkout закреплённого
engine, CMake, Xcode 26+ и iOS Metal toolchain. Незагруженный submodule и dirty или
неверный engine checkout не проходят проверку. На Linux CLI доступен для тестов,
но host отмечается `unsupported`; фиктивных Mac-замеров не создаётся.

Для workers сначала собрать и установить приложение из этой ветки обычным
`scripts/build-iphone.sh`. Открыть приложение на каждом устройстве и оставить
его foreground. Использовать уже установленную связь и адрес каждого control
port. CLI сам не ищет «первое» устройство, не запускает app и не меняет pairing.

```bash
./backcluster profile \
  --worker "ipad=${IPAD_CONTROL_IP}" \
  --worker "iphone=${IPHONE_CONTROL_IP}" \
  --timeout 3 \
  --output .backcluster/trio-profile.json
./backcluster doctor \
  --worker "ipad=${IPAD_CONTROL_IP}" \
  --worker "iphone=${IPHONE_CONTROL_IP}"
```

`IPAD_CONTROL_IP` и `IPHONE_CONTROL_IP` — локальные переменные с фактическими
адресами. По умолчанию control port 50061; для независимых локальных туннелей
можно указать `ipad=localhost:50161` и `iphone=localhost:50261`. IPv6 поддерживается
как `alias=[IPv6]:port`. Псевдонимы уникальны, `mac` зарезервирован. DNS-имена,
кроме localhost, не принимаются.

Один запрос на worker: `profile\n`. Ответ ограничен 16 KiB; общий deadline
включает отправку и все фрагменты чтения. EOF без newline, malformed JSON,
слишком большой ответ, недоступность и старый app дают явную ошибку данного
псевдонима. Успешные данные второго worker сохраняются. Raw ответы и адреса в
JSON и сообщения об ошибках не копируются. Старый `mem` не превращается в
новый профиль автоматически; нужно обновить app.

## Смысл профиля v1

`document_kind=backcluster_device_profile`, `schema_version=1`.
`captured_at` — время UTC, `hardware_validated=false` во всех снимках.
Host и workers содержат allowlist hardware/OS/runtime полей. Мобильный
`hardware.model` — product type, например `iPad17,1`, а не имя устройства.
Endpoint, UDID, serial, имя пользователя, signing team и пути в профиль не входят.

| Поле memory | Значение |
| --- | --- |
| `physical_bytes` | Физическая память устройства |
| `process_footprint_bytes` | Текущий footprint процесса worker |
| `process_available_bytes` | Текущая рекомендация `os_proc_available_memory()` для дополнительных аллокаций app |
| `metal_recommended_bytes` | Рекомендованный рабочий набор Metal, не дополнительная RAM |
| `metal_allocated_bytes` | Аллокации текущего Metal device |

Все размеры в целых байтах. `null` означает «не измерено», `0` сохраняется как
наблюдение. На Mac пока измеряется физическая память; app/Metal-поля остаются
`null`, поскольку Python collector не является процессом inference. Нельзя
считать остаток физической RAM подтверждённым inference budget.

Память app и Metal не складываем; значения динамические, это не разрешение
загружать модель. См. официальные описания
[os_proc_available_memory](https://developer.apple.com/documentation/os/os_proc_available_memory)
и [recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize).

Runtime worker сообщает app version/build, фактические compile-time версии
phone-attn, tail protocol и tail state format. Capabilities описывают функции
этой сборки, а не готовность конкретной модели/сессии. Engine commit и hashes
загруженных весов worker пока не удостоверяет: это следующий compatibility gate.

`control_round_trip_ns` — один sample полного control request, включая TCP
connect, вычисление профиля и JSON. Он не является latency бинарного inference
канала или payload throughput. `transport=unverified`: наличие сокета не
доказывает кабельную топологию. Оба payload links нужно измерить в BC-002.

## Проверки и exit codes

- `profile`: 0, если host macOS и все явно указанные workers опрошены; 1 при
  unsupported host, ошибке worker или I/O. Частичный snapshot всё равно выдаётся.
- `profile --check`: 0 для корректного формата, 1 для некорректного. Проверяет
  только снимок, не свежесть, forward, identity модели или readiness.
- `doctor`: 0, если все перечисленные проверки прошли; 1 при проблеме.
  `cluster_ready=false` даже при успешном doctor: model lock, бюджеты и forward
  ещё должны пройти отдельную приёмку.
- Ошибки синтаксиса аргументов: 2.

`--output` сохраняет JSON атомарно с permissions 0600. Локальные результаты в
`.backcluster/` игнорируются Git. Команды без `--output` печатают JSON в stdout.

## Проверка разработки

```bash
python3 -m unittest discover -s tests/backcluster -v
scripts/check-listeners.sh
# На macOS с Xcode и загруженным submodule:
tests/security/run.sh
```

Тесты используют настоящий локальный socket server: fragmented replies,
EOF, timeout/slow-drip, size bounds, старый app, malformed profiles, privacy
allowlist, null-значения и атомарную запись. CI запускает Python на Linux/macOS
и весь существующий security suite плюс Objective-C++ syntax check worker на macOS с Xcode 26.3 (upstream использует Metal 4). Это не сборка iOS-приложения и не
замеры скорости инференса; Apple build и тест на целевых устройствах обязательны.
