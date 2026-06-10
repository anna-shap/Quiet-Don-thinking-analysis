# Руководство пользователя

## 1. Подготовка окружения

Рекомендуется Python 3.10.

```bat
py -3.10 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -U pip "setuptools==70.2.0" wheel
python -m pip install -r requirements.txt
```

## 2. Подготовка корпуса

Полный текст романа должен находиться здесь:

```text
data/corpus/quiet_don.txt
```

Если полный корпус пока не подключён, можно проверить систему на sample:

```text
data/corpus/quiet_don_SAMPLE.txt
```

## 3. Проверка проекта

```bat
python -m tdmm.validate_project --config config.json
```

Для sample:

```bat
python -m tdmm.validate_project --config config.json --corpus data\corpus\quiet_don_SAMPLE.txt
```

## 4. Запуск анализа

Проверочный запуск:

```bat
python -m tdmm.run_pipeline --config config.json --corpus data\corpus\quiet_don_SAMPLE.txt --out out_demo_sample --make-plots --max-sents 50 --clean-out
```

Полный запуск:

```bat
python -m tdmm.run_pipeline --config config.json --out out_final --make-plots --clean-out
```

## 5. Где смотреть результаты

| Файл | Что смотреть |
|---|---|
| `STATS.xlsx` | основная статистика |
| `SHOWCASE.xlsx` | примеры для ВКР и доклада |
| `BEST_THESIS_EXAMPLES` внутри `SHOWCASE.xlsx` | лучшие примеры для текстового разбора |
| `PHRASE_QUALITY_REPORT.xlsx` | качество фразового словаря |
| `QUALITY_DASHBOARD.xlsx` | предупреждения и сомнительные случаи |
| `MANUAL_VERIFICATION.xlsx` | файл ручной проверки |
| `plot_*.png` | графики для Word/презентации |
| `plot_*.svg` | векторные графики |
| `RUN_METADATA.json` | параметры запуска и воспроизводимость |

## 6. Что проверять вручную

В первую очередь проверяются:

1. `MelekhovLink = PROBABLE`.
2. `PerspectiveConfidence = LOW`.
3. `Perspective = UNKNOWN`.
4. Частые `EXTENDED`-фразы.
5. Фразовые маркеры с риском шума.
6. Шумовые исключения из `noise_check`.

## 7. Типовые проблемы

Если нет `quiet_don.txt`, полный запуск не выполнится. Используйте sample-команду или положите полный корпус в `data/corpus/quiet_don.txt`.

Если не создаются Parquet-файлы, установите `pyarrow`. CSV и Excel при этом остаются основными рабочими форматами.

Если график отсутствует, проверьте, есть ли данные для соответствующей категории. Некоторые графики не создаются, если в sample нет нужных контекстов.
