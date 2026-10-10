# Продолжение

Контекст владельца: базовый MacBook Pro M4 Pro с 24 ГБ, iPhone 17 Pro Max, iPad Pro 13 M5 256 ГБ. Целевая модель **Qwen3.8-27B**; 3.5 была исправлена пользователем. Цель — распределённая локальная модель для coding workflow, включая настоящее распределение весов, а не только независимые агенты.

## Уже сделано

- Изучены parent и закреплённый engine; происхождение в `UPSTREAM.md`.
- Сохранены upstream code/history, license и оригинальный README.
- Подготовлены audit, architecture, memory analysis, roadmap, backlog и hardware runbook.
- Пример `profiles/apple-trio.example.json` является планом, не исполняемой конфигурацией.
- Репозиторий опубликован: https://github.com/art9762/backcluster.
- Начат BC-001: `backcluster doctor/profile`, read-only worker report, схема v1, socket regression tests и CI. Детали в [DIAGNOSTICS.md](DIAGNOSTICS.md).

## Не сделано

Нет нового inference scheduler, Apple build, inference benchmarks, проверки signing/Thunderbolt/background GPU, скачанных weights или проверенных model hashes. Мобильный control runtime дополнен командой `profile`; binary inference protocols не менялись. Доступность 48 ГБ как единого пространства памяти не заявляется. Отдельного downstream engine fork пока нет.

## Следующее действие

Собрать app с новым `profile` на Mac, выполнить `doctor` и собрать host/pair/trio snapshots по `DIAGNOSTICS.md`. Затем выполнить BC-001/002 из `HARDWARE_RUNBOOK.md` на устройствах и закрепить модель BC-003. После baseline BC-004 сравнить Mac+iPad с Mac+iPhone. Не начинать с kernel rewrite или speculative decode через три устройства.

При дальнейшей разработке читать актуальные repository instructions. Не менять pin движка случайно через submodule update remote. Перед первым engine change создать отдельный доступный fork и соблюдать его инструкции; коммиты parent и submodule публикуются отдельно.

## Ключевые решения

Два режима: accelerator хранит полную модель на Mac; capacity распределяет уникальные веса. Начальная роль iPad — compute, iPhone — KV, но её должен подтвердить benchmark. Цепочка трёх compute nodes требует middle-stage и state coordinator. Remote KV нельзя совмещать с текущим tail без доработки. Потеря единственной копии состояния требует ошибки/восстановления, а не тихого fallback.
