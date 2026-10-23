# Backlog

Статусы разделяют исполняемый software и аппаратную приёмку. BC-001/002/003/004/006/010/011/012/021/022/023/024/030/031 реализованы или подготовлены программно в указанном ниже объёме; real-device quality/performance/installation acceptance открыта. BC-026 имеет portable toolkit, но production native distributed restore и tool-agent integration ещё требуют разработки. См. [SOFTWARE_ACCEPTANCE.md](SOFTWARE_ACCEPTANCE.md) и [HARDWARE_RUNBOOK.md](HARDWARE_RUNBOOK.md). `P0` — обязательная основа, `P1` — основной результат, `P2` — оптимизация.

| ID | Приоритет | Зависит от | Работа и точки изменения | Проверяемый результат |
|---|---|---|---|---|
| BC-001 | P0 | — | **SOFTWARE:** doctor/profile, protocol/base/build compatibility и loaded artifact receipt | Код/tests; Apple gates в CI; измерения трёх узлов ждут железо |
| BC-002 | P0 | 001 | Cable payload bench, mobile signing и standalone app | RTT/throughput для обоих links, app budget без debugger, две независимые связи |
| BC-003 | P0 | — | Model/draft/shard lock и GGUF metadata validation | Pinned revision, bytes/hash, template, layer map; mismatch блокируется |
| BC-004 | P0 | 001,003 | Mac-only baseline, `serve.sh`, `bench/` | IQ4, 16k/32k/64k где помещается; одинаковые настройки, пять повторов |
| BC-005 | P0 | 002,004 | Mac+iPhone и Mac+iPad; split search | Лучший pair/split по wall time, проверено влияние SME и draft |
| BC-006 | P0 | 004 | **SOFTWARE:** SSE/JSON/reasoning/cancel и sandboxed read/edit/test tool fixture | Реальная модель и установленный OpenCode ещё проходят hardware/offline gate |
| BC-010 | P1 | 001,002 | **SOFTWARE:** explicit registry, live route и role selection | Порядок discovery не выбирает worker; hardware authentication не заявляется |
| BC-011 | P1 | 010,003 | **SOFTWARE:** independent budgets/reserves/peak-state admission на live launch | Unknown/mismatch/thermal/memory отказывают; target peaks ещё не измерены |
| BC-012 | P1 | 005,011 | **SOFTWARE:** explicit one/two-worker remote-KV launch и block-cyclic ownership estimates | Actual KV на обоих узлах и consistent real-model output ждут железо |
| BC-013 | P1 | 012,020 | Weighted page ownership, `ggml-metal-remote.mm` | Правильный position mapping, fetch/truncate/restore, heterogenous budgets |
| BC-020 | P0 | 003 | **SOFTWARE:** hash-locked patches поверх published base pin, content-addressed materializer | Clean clone воспроизводит patched source без недоступного gitlink; downstream published pin остаётся отдельным release шагом |
| BC-021 | P1 | 005,011 | **SOFTWARE:** strict split-decode/capacity launcher | Host пока открывает full GGUF; цель RAM Mac -4 GiB ещё не доказана |
| BC-022 | P1 | 020,021 | **SOFTWARE:** streaming shard splitter [start,end), global map и verified manifests | Exact trunk cover; tensors/layer arrays/MTP/head provenance проверяются |
| BC-023 | P1 | 022 | **SOFTWARE:** native middle residual protocol и iOS build integration | Bounded framing/source/range/epoch/request checks; Apple transport/actual model ждут железо |
| BC-024 | P1 | 023 | **SOFTWARE:** native Mac-relayed three-stage prompt/decode + CPU correctness/failure gate | Real llama_decode fixture; target model/Metal/quality/performance ещё не приняты |
| BC-025 | P2 | 012,024 | Совместить compute-shards и remote KV | Tail видит всю нужную историю; нет двойного учёта или пропуска attention |
| BC-026 | P1 | 024 | **PARTIAL:** BCTX toolkit, atomic checkpoint store, durable tool journal и explicit audited tool-result reconciliation | Portable replay/crash/reconciliation tests готовы; native distributed KV/GDN/RNG checkpoint и API-agent integration открыты |
| BC-027 | P2 | 026 | Distributed speculative decode/recurrent replay | Accept/reject откатывает все стадии; measured accepted-token gain |
| BC-028 | P1 | 011,024 | Thermal policy и безопасная смена профиля | Warning/serious/critical state обрабатываются без порчи сессии |
| BC-030 | P1 | 012,021 | **SOFTWARE:** CLI, fork-safe installer, distinct app identity и explicit cluster modes | Dry-run/live receipts; hardware standalone installation ждёт устройств |
| BC-031 | P1 | 006,030 | **SOFTWARE:** opencode-config с explicit installed client version | Реальный клиент, local tools/reasoning/cancel/offline loop ещё не приняты |
| BC-032 | P1 | 024,026,028,031 | Offline acceptance и release report | 10 задач, час нагрузки, fail tests, воспроизводимая установка и честный отчёт |

## Первая аппаратная сессия

1. BC-001/002: сохранить обезличенный профиль и raw link samples. Не менять kernels.
2. BC-003/004: artifact lock и baseline с существующим engine.
3. BC-005/006: выбрать pair и получить первый полезный coding loop.
4. BC-010/011: заполнить implemented registry/admission фактическими budgets/routes/identities.
5. BC-021/024: real-model correctness и освобождение RAM на парах, затем native pipeline; без distributed speculation.

Одна работа меняет один фактор. Каждый performance change сопровождается одинаковой моделью, настройками, командами, до/после и quality check. Новый protocol требует одинаковой совместимой сборки Mac/iOS. Engine changes публикуются как проверяемый pinned fork либо hash-locked project patches; локальный dirty submodule не является release source.

## Что не делать в начале

Не тратить первый этап на новый inference framework, свой редактор кода, облачный dashboard, обучение drafter, модели на сотни миллиардов параметров или background GPU workaround. Не фиксировать пропорции слоёв по одной только RAM. Не выдавать вспомогательный независимый LLM-agent на телефоне за distributed inference основной модели.

## Подготовлено до аппаратной границы, 2026-10-11

- BC-002: native payload echo + CLI raw/summary + concurrent workers. Signing, actual route/link mode и standalone budgets ещё не измерены.
- BC-003/022: artifact content lock, streaming shards, GGUF metadata/layout/source hash verification. Реальная модель/drafter и её engine support ещё не приняты.
- BC-010/011: explicit registry, live route/build/artifact checks и live admission управляют cluster launch; реальные peak-state estimates ещё измеряются.
- BC-021/024/030: executable pair/pipeline/remote-KV launch и separate BackCluster app. Real-model quality, Apple gates и hardware acceptance проверяются отдельно.
- BC-004/006/031: bounded SSE benchmark/comparison, quality/tool/cancel fixture и OpenCode config готовы; реальная модель и offline клиент ещё не приняты.
- BC-023: native middle residual extension воспроизводится из locked patches; CPU gate не заменяет Apple/target-model gate.
- BC-026: independent BCTX checkpoint/journal toolkit проверен через sockets/replay/crashes; production native checkpoint integration ещё открыта.
