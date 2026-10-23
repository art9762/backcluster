# Восстановление результата инструмента после сбоя

`TranscriptJournal` записывает `running` до вызова инструмента и `completed`
после сохранения результата. Если процесс завершился между этими границами,
повторный `execute` с тем же ID поднимает `ToolExecutionIndeterminate`.
Побочное действие уже могло произойти. Восстановление ниже сохраняет независимо
проверенный результат и позволяет продолжить с тем же ID без повторного вызова.

Это portable toolkit BC-026. Production native OpenAI server и OpenCode пока
не используют этот журнал автоматически; команды не подключают его к ним.

1. Остановить все процессы, выполняющие инструменты через этот live journal.
   Дождаться завершения запущенных инструментов и их дочерних процессов.
2. Посмотреть исходный вызов. Результат содержит приватные arguments и transcript
   data, поэтому сохранять его локально в ignored `.backcluster/`:

   ```bash
   ./backcluster journal-call --journal .backcluster/tools.sqlite \
     --call-id call_1 --output .backcluster/call-1.json
   ```

3. Независимо проверить реальное состояние файла/системы, на которую действовал
   инструмент. Если результат остаётся неизвестен, оставить вызов `running`.
   Команды не повторяют инструмент и не устанавливают его фактический исход.
4. Создать приватный `.backcluster/reconcile-1.json` с ровно пятью полями.
   `call_id`, `tool`, `arguments` должны совпадать с просмотренной записью;
   `result` — JSON-результат, который получит продолжение, а `evidence` —
   непустое описание независимой проверки:

   ```json
   {
     "call_id": "call_1",
     "tool": "edit",
     "arguments": {"path": "fixture.py"},
     "result": {"edited": true},
     "evidence": "После остановки процесса проверены содержимое fixture.py и ожидаемый diff."
   }
   ```

5. Сохранить результат:

   ```bash
   ./backcluster journal-reconcile --journal .backcluster/tools.sqlite \
     --request .backcluster/reconcile-1.json --output .backcluster/recovered-1.json
   ```

Обе команды требуют существующий SQLite journal. Не передавать им экспорт
`journal.bin` из inference checkpoint. Output нельзя записать поверх journal,
его SQLite auxiliary files, executor lock или request. Без `--output` приватная
запись выводится в stdout. Ошибки возвращают exit code 1, синтаксис — 2.

Результат, SHA-256 evidence и timestamp сохраняются одной FULL-synchronous
SQLite transaction. Повтор той же reconciliation идемпотентен; изменение
результата/evidence или уже нормально завершённого вызова отвергается. Нельзя
создать новый tool call через reconciliation. `execute` с тем же ID и request
после восстановления возвращает сохранённый результат; другие arguments/tool
отвергаются. Экспорт `snapshot()` включает evidence и имеет schema version 2.

Межпроцессный executor lock блокирует reconciliation, пока библиотечный callback
ещё выполняется. Он освобождается при завершении callback или смерти процесса.
Lock не отслеживает отделившиеся дочерние процессы и не доказывает исход
побочного действия: шаги остановки и независимой проверки обязательны.

SQLite schema v1 автоматически и атомарно обновляется до v2 с сохранением
messages и tool receipts. Старые версии BackCluster отвергают v2; не запускать
их одновременно с обновлением. Уже выполненные tool receipts не откатывать
вместе с inference state и не удалять для обхода `running`.

В Python тот же путь доступен явно:

```python
from backcluster.checkpoints import TranscriptJournal

with TranscriptJournal(journal_path, create=False) as journal:
    journal.reconcile(call_id, tool, arguments, observed_result,
                      evidence="How the outcome was independently verified")
```

Regression suite проверяет реальный `os._exit` после side effect, продолжение
после reopen, блокировку живого executor в другом процессе, atomic rollback
при ошибке audit write, миграцию v1 и отказ при изменённом request/evidence.
