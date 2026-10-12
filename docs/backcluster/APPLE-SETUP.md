# Запуск на Mac и установка на iPhone / iPad

Подготовка 12 октября 2026. Модель по просьбе владельца не загружается.
Для инструментов разрешено до 5 ГБ загрузок. `prepare-apple.sh` не скачивает веса.

## Сборка

Из корня BackCluster-source:

```bash
scripts/prepare-apple.sh
```

Скрипт ставит CMake и Metal Toolchain, получает закреплённый engine,
создаёт Python venv с numpy, собирает Mac server/quantizer и IPA, запускает
security suite. Полученный `ios/build/Backburner.ipa` предназначен для
AltStore 2.2+ и требует подписи Apple перед установкой. Это одно приложение
для iPhone и iPad. В приложении пока сохранено имя Backburner.

## Установка через Xcode

Открыть `ios/Backburner/Sidecar.xcodeproj`, в Xcode → Settings → Accounts
добавить Apple ID. Подключить устройство кабелем, разблокировать, подтвердить
Trust и включить Developer Mode с перезагрузкой. Данные подписи задавать
только в терминале, не добавлять в git.

```bash
xcrun devicectl list devices
DEVELOPMENT_TEAM=YOUR_TEAM UDID=YOUR_DEVICE scripts/install-apple-device.sh
```

Повторить отдельно для iPhone и iPad: provisioning profile должен включать
каждое устройство. Бесплатная подпись обычно требует обновления каждые 7 дней.
Альтернативный путь установки IPA через AltStore описан в `docs/INSTALL-IPHONE.md`.

## Первый запуск Mac (после отдельной загрузки модели)

Модель: `~/Models/Qwen3.8-27B-IQ4_XS.gguf`, размер 15475951328 bytes,
SHA-256 `4b927360fa7c4aa41a734302f0795ca45df0e8b8a258bc76cc5fbf9d98a484db`.
Publisher: `bartowski/Qwen3.8-27B-GGUF`, revision
`0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a`. Размер и hash проверены по
метаданным publisher, локальные веса отсутствуют.

```bash
scripts/start-mac.sh
# В другом терминале:
curl -f http://127.0.0.1:8080/health
curl -f http://127.0.0.1:8080/v1/models
curl -f http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Ответь одним словом: привет"}],"max_tokens":32,"temperature":0}'
```

Начальный профиль: Mac, 16k context, draft off, mmap, proxy off. Глобальные
лимиты GPU не меняет. Остановка: Ctrl-C. Без веса inference не проверен.

## Подключение worker (после появления веса)

Подключать по одному устройству: существующие upstream helpers выбирают
первый найденный worker, это не готовый планировщик трёх compute-узлов.

```bash
source .venv/bin/activate
python scripts/split-gguf.py ~/Models/Qwen3.8-27B-IQ4_XS.gguf \
  ~/Models/tail-iq4xs-L40-nohead.gguf -L 40 --no-head
scripts/phone-tail.sh L40
PHONE=auto PHONE_ANE=0 DRAFT= SPEC_TYPE=none SPEC_REPLAY=0 \
  SME=0 MM_SME=0 PROXY=0 LOAD_MODE=mmap CTX=16384 scripts/serve.sh
```

Открыть Backburner на устройстве, разрешить Local Network и держать foreground.
Проверить, что `phone-tail.sh` подтвердил L40 и server сообщил split prefill.
Первый тест L40 не гарантирует оптимального использования памяти iPad.
Проверить каждое устройство отдельно по HARDWARE_RUNBOOK.md. Одновременное
распределение весов между Mac+iPhone+iPad ещё не реализовано.
