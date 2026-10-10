# Продолжение

Контекст владельца: базовый MacBook Pro M4 Pro с 24 ГБ, iPhone 17 Pro Max, iPad Pro 13 M5 256 ГБ. Целевая модель **Qwen3.8-27B**; 3.5 была исправлена пользователем. Цель — распределённая локальная модель для coding workflow, включая настоящее распределение весов, а не только независимые агенты.

## Уже сделано

- Изучены parent и закреплённый engine; происхождение в `UPSTREAM.md`.
- Сохранены upstream code/history, license и оригинальный README.
- Подготовлены audit, architecture, memory analysis, roadmap, backlog и hardware runbook.
- Пример `profiles/apple-trio.example.json` является планом, не исполняемой конфигурацией.
- Репозиторий опубликован: https://github.com/art9762/backcluster.
- Начат BC-001: `backcluster doctor/profile`, read-only worker report, схема v1, socket regression tests и CI. Детали в [DIAGNOSTICS.md](DIAGNOSTICS.md).

- Добавлены GGUF content lock, registry/compatibility, отдельные budget estimates, bounded link payload bench, Mac build/serve и SSE benchmark.
- Сборка BackCluster имеет отдельные bundle ID/display name/IPA; framework identity публикуется worker из build header.
- Полный software/hardware boundary и команды: [PRE_HARDWARE.md](PRE_HARDWARE.md).

## Не сделано

Нет нового inference scheduler, замеров inference на целевом железе, проверки signing/Thunderbolt/background GPU, скачанных weights или проверенного фактического model lock. Apple compilation/archive проверяются в CI; это не аппаратная приёмка. Мобильный control runtime дополнен командой `profile`; binary inference protocols не менялись. Доступность 48 ГБ как единого пространства памяти не заявляется. Отдельного downstream engine fork пока нет.

## Следующее действие

Начать с [PRE_HARDWARE.md](PRE_HARDWARE.md): установить новый app на iPad/iPhone, выполнить `doctor`, собрать host/pair/trio snapshots, payload link samples и Mac-only SSE baseline. Затем выполнить BC-001/002 из `HARDWARE_RUNBOOK.md` на устройствах и закрепить модель BC-003. После baseline BC-004 сравнить Mac+iPad с Mac+iPhone. Не начинать с kernel rewrite или speculative decode через три устройства.

При дальнейшей разработке читать актуальные repository instructions. Не менять pin движка случайно через submodule update remote. Перед первым engine change создать отдельный доступный fork и соблюдать его инструкции; коммиты parent и submodule публикуются отдельно.

## Ключевые решения

Два режима: accelerator хранит полную модель на Mac; capacity распределяет уникальные веса. Начальная роль iPad — compute, iPhone — KV, но её должен подтвердить benchmark. Цепочка трёх compute nodes требует middle-stage и state coordinator. Remote KV нельзя совмещать с текущим tail без доработки. Потеря единственной копии состояния требует ошибки/восстановления, а не тихого fallback.
