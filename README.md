# BackCluster

> Локальный стек для запуска **Qwen3.8-27B** на связке Apple-устройств: MacBook Pro M4 Pro · iPad Pro 13 M5 · iPhone 17 Pro Max.

[![Статус: исследование](https://img.shields.io/badge/status-research-blue)](docs/backcluster/ROADMAP.md)
[![Лицензия: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

BackCluster — downstream-проект на базе [StayLameBro/backburner](https://github.com/StayLameBro/backburner). Репозиторий объединяет исходный runtime, зафиксированную версию `llama.cpp` и план распределённого запуска модели между Mac, iPad и iPhone.

> **Текущий статус · 2026-10-09**  
> Это этап исследования и проектирования. Runtime пока не изменён, новый планировщик не реализован, а сценарии на целевых устройствах ещё не проверялись. Документы ниже описывают проверяемый план, а не готовые возможности.

## Зачем это нужно

Цель — получить один локальный OpenAI-compatible endpoint для инструментов программирования, сохранив на Mac память для редактора и сборок. Для этого проект последовательно проверяет:

- базовый запуск только на Mac;
- подключение iPhone и iPad по отдельности;
- распределение prefill, decode и контекста по реальной скорости сети и доступной памяти;
- восстановление сессии при отключении устройства;
- работу после установки без постоянного доступа к интернету.

Физические 24 + 16 + 12 ГБ нельзя считать единым доступным пулом: ОС, GPU, runtime и связь между устройствами съедают часть бюджета. Поэтому каждая цифра в проекте должна подтверждаться измерением.

## Быстрый старт

```bash
git clone <url-репозитория> backcluster
cd backcluster
git submodule update --init --recursive
```

Затем выполните upstream baseline по [оригинальному руководству](README.upstream.md). Скрипты и команды upstream сохранены без переименований; отдельного `backcluster serve` пока нет.

Профиль будущей конфигурации находится в [`profiles/apple-trio.example.json`](profiles/apple-trio.example.json). Сейчас это спецификация интерфейса для планировщика, а не конфигурация, которую читает runtime.

## Карта проекта

| Раздел | Назначение |
| --- | --- |
| [`docs/backcluster/AUDIT.md`](docs/backcluster/AUDIT.md) | Что уже есть в upstream и где нужны изменения |
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
