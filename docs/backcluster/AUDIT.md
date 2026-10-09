# Разбор backburner

Проверены исходники parent и engine на SHA из [UPSTREAM.md](../../UPSTREAM.md). Это статический анализ; Apple runtime здесь не запускался.

## Вывод

База подходит: Qwen3.8-27B, native Apple GPU, OpenAI-compatible server, split-prefill, remote KV и ограниченный split-decode уже существуют. Основная работа BackCluster — планирование разнородных устройств, распределение весов и надёжность состояния. Переписывать движок с нуля не требуется.

## Что подтверждено кодом

| Возможность | Реализация | Граница |
|---|---|---|
| API на Mac | `scripts/serve.sh`, `scripts/proxy.py` | Loopback, одна последовательность `-np 1`, SSD prefix cache |
| Два вычислительных узла при prefill | `llama.cpp/src/llama-split.cpp`, `tools/split-prefill/tail-server.h` в engine | Один `LLAMA_SPLIT_TAIL`; полная модель остаётся на Mac |
| Контекст на нескольких workers | `llama-kv-cache.cpp`, `ggml-metal-remote.mm`, `phone-attn/` | Список endpoints, циклическое распределение блоков по 4096 позиций |
| Разделение весов на два узла | `SPLIT_DECODE_L` в `serve.sh`, split-decode в engine | Один tail с output head; без drafter, speculation, proxy/checkpoints и remote KV |
| iOS/iPadOS worker | `RPCBridge.mm`, Swift UI | Native Metal; entitlement в проекте не гарантирует фактический лимит после установки |
| Профиль устройства | `bench/device-probe.py`, RPC `mem` | Hardware, memory, thermal, RTT; автоматического planner нет |
| Защита связи | `CableOnly.h`, `Tunnel.swift`, `WifiTunnel.swift` | Кабельные проверки, Wi-Fi через paired authenticated tunnel |
| Измерения | `launch-quality.py`, `turn-bench.py`, `session-bench.py`, `long-bench.py` | Нужны реальные устройства и одинаковые артефакты |

Внутреннее название `qwen35` в engine совместимо с архитектурной базой Qwen3.8 и не означает выбор модели 3.5.

## Ограничения и несогласованности

1. **Цепочки трёх вычислительных узлов нет.** `LLAMA_SPLIT_TAIL` принимает один endpoint. `TWO-PHONES.md` отдельно помечает вариант B с двумя tail workers как нереализованный. Разделение старого KV между workers уже есть.
2. **Проверка двух workers описана неодинаково.** README пишет tested, специализированный `TWO-PHONES.md` описывает два loopback-сервера и отсутствие проверки на двух реальных телефонах. Принимаем более узкое подтверждение: код есть, целевая тройка не проверена.
3. **Splitter умеет только хвост.** `scripts/split-gguf.py` имеет `-L`, но не конечный слой. Утверждение design-doc про любой диапазон не соответствует CLI. Для middle-stage нужны `[start,end)`, global/local layer IDs и обработка per-layer arrays/MTP.
4. **Remote KV отключает split-prefill.** В `llama-split.cpp` есть явный guard: tail не видит удалённые ключи. Одновременно считать слои на iPad и старый контекст на iPhone на большой глубине пока нельзя просто включением флагов.
5. **Бюджет памяти не индивидуален.** `serve.sh` вычисляет wired cap по первому устройству, использует его для следующего и выбирает минимальную равную долю. Reserve приложения по умолчанию 512 MiB. Для разных устройств нужен отдельный admission check.
6. **Remote KV специализирован.** `llama_kv_cache::remote_attach` требует head dimension 256, GQA ratio 6, одинаковые K/V rows, один stream и допустимый KV dtype. Qwen3.8-27B соответствует этим параметрам; другую архитектуру нельзя подключать только заменой пути.
7. **Mac baseline может отличаться.** Комментарии `serve.sh` описывают tuning на M4 Pro с 10 performance cores, пользователь указал базовый M4 Pro. Нужны фактические P/E/GPU cores и проверка конкуренции с компилятором.
8. **Split-decode имеет издержки.** Возвращает весь vocab в float32, около 1 MB на токен. Recurrent state нельзя произвольно откатить; при расхождении префикса текущий режим делает полный prefill.
9. **Единственная копия состояния не допускает тихого fallback.** Remote-KV engine останавливается при потере worker. Split-decode не может продолжать на Mac, который не держит все веса. Prefill accelerator может повторить batch на Mac, пока состояние полностью восстанавливаемо.
10. **State format привязан к engine.** Tail использует `PROTO_VERSION=2`, `STATE_FORMAT=1` и внутреннюю сериализацию llama.cpp. Shape checks не доказывают идентичность весов: нужны hashes и согласованные сборки.
11. **Автообнаружение не registry.** `phone-up.sh` выбирает начальный USB identifier и ищет адреса. Нужна устойчивая связь физического устройства, интерфейса, endpoint и роли; первая строка ответа для этого недостаточна.
12. **Установщик остаётся upstream.** Update URL, приложение, AltStore feed и пути ещё Backburner. Это отдельная задача ребрендинга/упаковки.

## Переиспользуем и расширяем

Сохраняем native worker, transport, llama server, GGUF loader, kernels, state-sync, partial-attention merge, бенчмарки и listener/commit checks. Добавляем registry, planner, per-device budgets, middle-stage, корректное восстановление, thermal policy, установку и OpenCode offline-профиль.

Ускорение engine против stock llama.cpp, эффект iPad, эффект iPhone, drafter, KV dtype и prefix cache измеряем отдельно. Иначе выигрыш от прогретого кэша или другой точности легко ошибочно назвать ускорением от трёх чипов.
