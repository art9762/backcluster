# Модель, память и связь

Цель: **Qwen3.8-27B**, text-only coding на первом этапе. Официальная конфигурация и snapshot revisions перечислены в [UPSTREAM.md](../../UPSTREAM.md).

## Модельный контракт

У 27B — 64 trunk-слоя, hidden size 5120, 16 full-attention слоёв между Gated DeltaNet, 24 Q-heads и 4 KV-heads размером 256; native context 262144. Это объясняет требования текущего remote-KV backend. Максимум из model card не является проверенным рабочим контекстом нашего кластера.

| Артефакт | Размер файла по таблице publisher, GB | Назначение |
|---|---:|---|
| IQ4_XS | 15.48 | Первый baseline и accelerator |
| Q4_K_M | 17.44 | Сравнение качества, если помещается с runtime и IDE |
| Q5_K_M | 20.92 | Первый кандидат для capacity-mode |
| Q6_K | 23.86 | Следующий эксперимент после устойчивого распределения весов |
| Q8_0 | 29.12 | Не начальная цель; сначала доказать бюджет каждого узла |

Источник размеров: [автор GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF), состояние 2026-10-09. GB здесь десятичные, GiB в measurements — `bytes / 2^30`. Размер файла не равен resident memory: добавляются KV, recurrent state, allocator, compute buffers, draft и возможные реплики тензоров. Эти файлы имеют новую раскладку квантизации; совместимость закреплённого engine проверяется первым запуском.

DFlash2 берём из `z-lab/Qwen3.8-27B-DFlash2` и преобразуем соответствующим upstream script. Фиксируем исходную revision, команду, target hash и результат преобразования. Сначала benchmark без drafter, затем с ним. Не переносим draft другой модели, не предполагаем поддержку distributed rollback.

## Физическая память и admission control

| Устройство | Физическая RAM | Что нужно измерить |
|---|---|---|
| Базовый MacBook Pro M4 Pro | 24 ГБ со слов владельца | CPU/GPU cores, свободная RAM, memory pressure, swap delta, Metal working set при открытом IDE |
| iPad Pro 13 M5, 256 ГБ | 12 ГБ по спецификации Apple | Реальный app budget после подписи, footprint, working set, нагрев, доступные Metal capabilities |
| iPhone 17 Pro Max | Ожидаемые 12 ГБ по конфигурации владельца; подтвердить probe | Физическая память и фактический лимит установленного приложения, shared buffers, нагрев |

Не используем сумму 48 как бюджет модели. Для каждого узла отдельно:

`weights + unique_state + replicas + KV + recurrent + compute_peak + staging + draft + reserve <= allowed_peak`

`allowed_peak` определяется одновременно OS/app, GPU и текущим состоянием системы. `os_proc_available_memory()` сообщает оставшийся запас приложения; это не полный лимит и не сумма с физической RAM. Измеряем до/после загрузки и на пике, а не только при startup. Нельзя дважды списывать reserve или дважды учитывать одни и те же shared CPU/GPU buffers.

Начальные **предположения для эксперимента**, не обещание доступности: на Mac оставить 6–8 GiB под ОС, редактор и сборку; на каждом mobile-worker оставить не менее 1 GiB сверх измеренного пика runtime либо больший эмпирический запас. Итоговые allocation limits неизвестны до профиля. Не форсируем общий wired limit «под все 24 ГБ».

Для capacity-mode сначала добиться уменьшения peak-памяти Mac хотя бы на 4 GiB относительно той же модели/контекста в accelerator baseline. Это целевой критерий проекта, а не уже полученный результат.

## KV и recurrent state

Для текущей архитектуры upstream считает на token по всем full-attention слоям:

| KV dtype | Bytes/token | 65536 tokens, GiB |
|---|---:|---:|
| q8_0 | 34816 | 2.125 |
| q4_0 | 18432 | 1.125 |
| f16 | 65536 | 4.000 |

Это только KV: GDN state, checkpoints, draft cache и buffers считаются отдельно. В accelerator-mode часть состояния может дублироваться на Mac и tail. В capacity-mode budget вычисляется для конкретного layer range. Полное распределённое сохранение контекста временно создаёт дополнительные host/transfer buffers; их peak тоже входит в admission check.

Рабочая последовательность: 16k/32k baseline, 64k после проверки памяти, затем 96k/128k remote KV. 196k/262k не критерий первого релиза.

## Каналы и payload

iPhone имеет USB 3 до 10 Gb/s. iPad поддерживает Thunderbolt/USB4 до 40 Gb/s и USB 3 до 10 Gb/s. Это возможности портов, не измеренная полоса TCP-сокета Backburner. Наличие Thunderbolt не доказывает работоспособный Thunderbolt Bridge между этим Mac и iPad.

MVP использует поддержанный кабельный IP transport. Проверяем negotiated link, payload throughput и RTT на реальных сообщениях; общий hub и два отдельных порта сравниваем отдельно. Проверка offline проводится при выключенных внешних сетях, но с сохранённой кабельной сетью.

Для residual в f16: `256 * 5120 * 2 = 2621440 bytes` на 256-token chunk; один decode residual — 10240 bytes. Текущий split-decode может передавать f32 residual (вдвое больше) и около 993280 bytes logits обратно. Учитываем фактический wire dtype, relay, copies и sync — делить payload на паспортные 40 Gb/s недостаточно.

Суммарные TOPS трёх чипов не дают ожидаемый tok/s. Dense decode ограничен памятью, последовательностью стадий и latency; большой prefill имеет другой профиль.

## Мобильная ОС

Рабочий baseline требует foreground worker и незаблокированный экран. Не делаем безусловное утверждение, что background GPU невозможен на всех новых ОС: у Apple есть отдельный continued-processing GPU API с entitlement и проверкой поддержки. Он не реализован в этой схеме и не является условием успеха MVP. Даже его наличие не гарантирует бессрочный сервер с заблокированным экраном.

Проверяем именно release/sideloaded app: debugger и signing могут менять доступную память. M5/iPhone без активного охлаждения требуют длительного thermal-теста. Больше RAM на бумаге не устраняет thermal throttling или завершение приложения ОС.
