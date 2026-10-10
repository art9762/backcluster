# BackCluster

> Локальный стек для запуска **Qwen3.8-27B** на связке Apple-устройств: MacBook Pro M4 Pro · iPad Pro 13 M5 · iPhone 17 Pro Max.

[![Статус: P1 в разработке](https://img.shields.io/badge/status-P1_in_progress-blue)](docs/backcluster/ROADMAP.md)
[![Лицензия: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

BackCluster — downstream-проект на базе [StayLameBro/backburner](https://github.com/StayLameBro/backburner). Репозиторий объединяет исходный runtime, зафиксированную версию `llama.cpp` и план распределённого запуска модели между Mac, iPad и iPhone.

> **Текущий статус · 2026-10-10**
> Начат P1: работают CLI `doctor`/`profile`, проверка формата снимков и новый read-only профиль мобильного worker. Есть socket regression tests и CI. Планировщик и распределение весов ещё не реализованы; Apple build и сценарии на целевых устройствах пока не проверены.

## Зачем это нужно

Цель — получить один локальный OpenAI-compatible endpoint для инструментов программирования, сохранив на Mac память для редактора и сборок. Для этого проект последовательно проверяет:

- базовый запуск только на Mac;
- подключение iPhone и iPad по отдельности;
- распределение prefill, decode и контекста по реальной скорости сети и доступной памяти;
- восстановление сессии при отключении устройства;
- работу после установки без постоянного доступа к интернету.

Физические 24 + 12 + 12 ГБ нельзя считать единым доступным пулом: ОС, GPU, runtime и связь между устройствами съедают часть бюджета. Реальную RAM каждого устройства фиксируем по измерению, не по ожидаемым характеристикам.

## Быстрый старт

```bash
git clone https://github.com/art9762/backcluster.git
cd backcluster
git submodule update --init --recursive
./backcluster doctor
./backcluster profile --output .backcluster/host-profile.json
```

Затем выполните upstream baseline по [оригинальному руководству](README.upstream.md). Скрипты и команды upstream сохранены без переименований; отдельного `backcluster serve` пока нет.

Сбор профиля iPad/iPhone, схема памяти, ограничения и exit codes описаны в [руководстве по диагностике](docs/backcluster/DIAGNOSTICS.md). Python 3.9+, без дополнительных пакетов.

Профиль будущей конфигурации находится в [`profiles/apple-trio.example.json`](profiles/apple-trio.example.json). Сейчас это спецификация интерфейса для планировщика, а не конфигурация, которую читает runtime.

## Карта проекта

| Раздел | Назначение |
| --- | --- |
| [`docs/backcluster/AUDIT.md`](docs/backcluster/AUDIT.md) | Что уже есть в upstream и где нужны изменения |
| [`docs/backcluster/DIAGNOSTICS.md`](docs/backcluster/DIAGNOSTICS.md) | Работающие doctor/profile и сбор снимков устройств |
| [`docs/backcluster/ARCHITECTURE.md`](docs/backcluster/ARCHITECTURE.md) | Роли устройств, режимы, протокол и восстановление |
| [`docs/backcluster/MODEL_AND_MEMORY.md`](docs/backcluster/MODEL_AND_MEMORY.md) | Квантизации и бюджет памяти |
| [`docs/backcluster/ROADMAP.md`](docs/backcluster/ROADMAP.md) | Этапы, зависимости и критерии готовности |
| [`docs/backcluster/BACKLOG.md`](docs/backcluster/BACKLOG.md) | Разбивка работы на задачи BC-* |
| [`docs/backcluster/HARDWARE_RUNBOOK.md`](docs/backcluster/HARDWARE_RUNBOOK.md) | Подготовка устройств и сценарии измерений |
| [`docs/backcluster/NEXT_SESSION.md`](docs/backcluster/NEXT_SESSION.md) | Точка входа для следующей рабочей сессии |
| [`UPSTREAM.md`](UPSTREAM.md) | Происхождение кода и лицензии |
| [`README.upstream.md`](README.upstream.md) | Оригинальная документация backburner |

## Что считается готовым

Проект не объявляет режим рабочим по одному успешному запуску. Для каждой функции нужны:

1. воспроизводимый сценарий;
2. логи и измерения памяти, задержки и пропускной способности;
3. повтор после отключения одного из устройств;
4. проверка на реальной coding-задаче;
5. описание ограничений в соответствующем документе.

До появления этих данных распределение весов, выигрыш от тройки устройств и устойчивость к отказам считаются **неподтверждёнными**.

## Разработка

`llama.cpp/` подключён как submodule на фиксированном commit. Изменения engine следует делать в отдельном downstream fork и только затем обновлять gitlink в этом репозитории.

Для локальной работы сначала воспроизведите baseline upstream. Новый код и измерения добавляйте вместе с обновлением документации и критерия готовности в backlog.

## Лицензия и происхождение

Основной репозиторий распространяется по [MIT](LICENSE). Исходный проект — [backburner](https://github.com/StayLameBro/backburner). Веса моделей, ключи и данные для подписи приложения в этот репозиторий не входят.
