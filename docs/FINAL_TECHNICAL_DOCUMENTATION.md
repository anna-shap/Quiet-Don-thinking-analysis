# Финальная техническая документация TD-MindMarkers

## 1. Паспорт проекта

| Параметр | Значение |
|---|---|
| Название | TD-MindMarkers |
| Тип | Python-пайплайн корпусно-филологического анализа |
| Материал | роман М. А. Шолохова «Тихий Дон» |
| Основной герой | Григорий Мелехов |
| Предмет анализа | маркеры мыслительной деятельности |
| Подход | правиловой NLP: словари, лемматизация, фразовые шаблоны, эвристики |
| Основной профиль | `DIPLOMA_STRICT` |
| Основные результаты | CSV/Excel/PNG/SVG/JSON |

## 2. Смысл разработки

Проект предназначен для того, чтобы перейти от ручного выборочного анализа отдельных цитат к воспроизводимой корпусной процедуре. Он извлекает маркеры мыслительной деятельности, связывает их с контекстами Григория Мелехова и создаёт статистические и визуальные материалы для ВКР.

## 3. Архитектура

```mermaid
flowchart TD
    A[quiet_don.txt] --> B[Выделение заголовков]
    B --> B1[HEADINGS.csv]
    B --> C[Очищенный аналитический текст]
    C --> D[Сегментация на предложения]
    D --> E[Токенизация]
    E --> F[Лемматизация]

    G[s1_markers.csv] --> J[MARKERS_MASTER]
    H[alexandrova_extract.txt] --> J
    I[phrases.csv] --> J
    M[microthemes.csv] --> J

    J --> K[WORD-поиск]
    J --> L[PHRASE-поиск]
    F --> K
    F --> L

    K --> N[TD_HITS]
    L --> N
    O[exclusions.csv] --> N

    N --> P[Фильтрация шума]
    P --> Q[WORD внутри PHRASE не считается]
    Q --> R[TD_HITS_TAGGED]
    R --> S[MelekhovLink]
    R --> T[Perspective]
    R --> U[Part / Book]

    U --> V[STATS.xlsx]
    U --> W[SHOWCASE.xlsx]
    U --> X[PHRASE_QUALITY_REPORT.xlsx]
    U --> Y[QUALITY_DASHBOARD.xlsx]
    U --> Z[MANUAL_VERIFICATION.xlsx]
    U --> AA[Графики]
    U --> AB[RUN_METADATA.json]
```

## 4. Модель данных

```mermaid
erDiagram
    MARKERS_MASTER ||--o{ TD_HITS : marker_id
    SENTS ||--o{ TD_HITS : sent_id
    TD_HITS ||--|| TD_HITS_TAGGED : hit_id
    PARTS ||--o{ TD_HITS_TAGGED : sent_id

    MARKERS_MASTER {
        string marker_id
        string marker_label
        string microtheme
        string phrase_family
        string unit_type
        string unit_raw
        string unit_norm
        string precision_level
        string source_type
    }

    SENTS {
        int sent_id
        string sent_text
        int word_count
    }

    TD_HITS {
        int hit_id
        string hit_type
        string marker_id
        int sent_id
        int token_idx
        int token_end_idx
        string target_surface
        string target_lemma
        bool is_noise
        bool count_in_main_stats
    }

    TD_HITS_TAGGED {
        int hit_id
        string MelekhovLink
        string MelekhovEvidence
        string Perspective
        string PerspectiveEvidence
        string PerspectiveConfidence
        string SpeakerDetected
        string AddresseeDetected
        string Part
        string Book
    }
```

## 5. Словарная база

Файл `phrases.csv` содержит 686 фразовых маркеров. Каждый маркер имеет микротему, семейство, уровень точности и комментарий о возможном риске шума.

Ключевые микротемы:

- процесс размышления;
- инициация мысли;
- память / воспоминание;
- понимание / ясность;
- решение / намерение;
- сомнение / колебание;
- внутренняя речь;
- догадка / соображение;
- навязчивая мысль;
- голова / ум как орган мышления;
- побуждение к мышлению;
- эмоционально окрашенная мысль.

## 6. Алгоритм извлечения

WORD-маркеры ищутся по нормальной форме слова. PHRASE-маркеры ищутся по последовательности лемм с допустимым разрывом `phrase_gap = 2`. При вложенных фразах используется режим `prefer_longest`. Это позволяет избежать ситуации, когда `ему стало ясно` и `стало ясно` одновременно увеличивают частоту одного и того же когнитивного события.

## 7. Привязка к Григорию Мелехову

| Метка | Значение |
|---|---|
| `YES` | имя героя найдено в том же предложении |
| `PROBABLE` | имя героя найдено в соседнем предложении |
| `NO` | связь с героем не обнаружена |

Поле `MelekhovEvidence` показывает основание разметки.

## 8. Речевая перспектива

| Категория | Интерпретация |
|---|---|
| `HERO_INNER_SPEECH` | внутренняя речь героя |
| `HERO_DIRECT_SPEECH` | прямая речь героя |
| `NARRATOR_ABOUT_HERO` | повествователь сообщает о мышлении героя |
| `OTHERS_TO_HERO` | другой персонаж обращается к герою или побуждает его думать |
| `OTHERS_ABOUT_HERO` | другой персонаж говорит о герое |
| `OTHERS_NEAR_HERO` | чужая речь рядом с героем |
| `UNKNOWN` | перспектива не определена |

Финальная версия усиливает поверхностную атрибуцию говорящего и адресата через поля `SpeakerDetected` и `AddresseeDetected`.

## 9. Отчёты

`STATS.xlsx` содержит основные таблицы: частотность, покрытие, микротемы, перспективы, плотность, индексы, семейства фраз и матрицы.

`SHOWCASE.xlsx` содержит готовые примеры для ВКР, включая `BEST_THESIS_EXAMPLES`.

`PHRASE_QUALITY_REPORT.xlsx` показывает качество фразового словаря.

`QUALITY_DASHBOARD.xlsx` собирает предупреждения и сомнительные случаи.

`MANUAL_VERIFICATION.xlsx` предназначен для ручной экспертной проверки.

## 10. Аналитические индексы

Введены интерпретируемые показатели:

- когнитивная плотность: число засчитанных маркеров на 10000 слов;
- индекс внутренней речи: доля `HERO_INNER_SPEECH` среди связанных контекстов;
- индекс внешнего когнитивного давления: доля чужой речи о/к герою;
- индекс расширенной когнитивности: доля `EXTENDED` среди `CORE + EXTENDED`.

Эти показатели не являются психологическими измерениями личности героя. Они являются формальными корпусными индикаторами распределения языковых маркеров.

## 11. Ограничения

Главное ограничение — правиловой характер анализа. Система не понимает художественный текст как человек. Она выделяет формальные маркеры и применяет объяснимые эвристики. Поэтому итоговая интерпретация должна опираться на ручную проверку таблиц `MANUAL_VERIFICATION.xlsx`, `QUALITY_DASHBOARD.xlsx` и `SHOWCASE.xlsx`.
