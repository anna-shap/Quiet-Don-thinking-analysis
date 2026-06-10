# Changelog финальной версии

## Финальная итерация

- Обновлена документация под фактическое состояние проекта.
- `phrases.csv`: 686 фразовых маркеров.
- Введены и сохранены семейства фраз `phrase_family`.
- Нормализованы микротемы через `microthemes.csv`.
- Заголовки корпуса выделяются в `HEADINGS.csv`.
- Реализован `prefer_longest` для вложенных фраз.
- WORD внутри PHRASE не учитывается в основной статистике.
- Усилена атрибуция говорящего/адресата.
- Добавлен `MANUAL_VERIFICATION.xlsx`.
- Добавлен `BEST_THESIS_EXAMPLES`.
- Добавлены `melekhov_profile` и `analytical_indices`.
- Добавлен `plot_melekhov_cognitive_profile`.
- `RUN_METADATA.json` переведён на относительные пути.
- Тесты проходят: `2 passed`.
