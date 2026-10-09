# Продолжение

Контекст владельца: базовый MacBook Pro M4 Pro с 24 ГБ, iPhone 17 Pro Max, iPad Pro 13 M5 256 ГБ. Целевая модель **Qwen3.8-27B**; 3.5 была исправлена пользователем. Цель — распределённая локальная модель для coding workflow, включая настоящее распределение весов, а не только независимые агенты.

## Уже сделано

- Изучены parent и закреплённый engine; происхождение в `UPSTREAM.md`.
- Сохранены upstream code/history, license и оригинальный README.
- Подготовлены audit, architecture, memory analysis, roadmap, backlog и hardware runbook.
- Пример `profiles/apple-trio.example.json` является планом, не исполняемой конфигурацией.

## Не сделано

Нет новых runtime features, Apple build, inference benchmarks, проверки signing/Thunderbolt/background GPU, скачанных weights или проверенных model hashes. Доступность 48 ГБ как единого пространства памяти не заявляется. Отдельного downstream engine fork пока нет. Удалённую публикацию parent проверить по фактическому git remote; сам этот документ её не подтверждает.

## Следующее действие

Прочитать `HARDWARE_RUNBOOK.md`, выполнить BC-001/002 на устройствах и закрепить модель BC-003. После baseline BC-004 сравнить Mac+iPad с Mac+iPhone. Не начинать с kernel rewrite или speculative decode через три устройства.

При дальнейшей разработке читать актуальные repository instructions. Не менять pin движка случайно через submodule update remote. Перед первым engine change создать отдельный доступный fork и соблюдать его инструкции; коммиты parent и submodule публикуются отдельно.

## Ключевые решения

Два режима: accelerator хранит полную модель на Mac; capacity распределяет уникальные веса. Начальная роль iPad — compute, iPhone — KV, но её должен подтвердить benchmark. Цепочка трёх compute nodes требует middle-stage и state coordinator. Remote KV нельзя совмещать с текущим tail без доработки. Потеря единственной копии состояния требует ошибки/восстановления, а не тихого fallback.
