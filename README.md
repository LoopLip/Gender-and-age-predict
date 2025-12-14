# Age-Gender-Estimation

## 1. О проекте
Проект для предсказания возраста и пола по фотографии. Включает:
- обучение модели (age + gender),
- инференс по изображению/видео,
- утилиты диагностики и демонстрационные скрипты.

## 2. Быстрый старт
Скачайте файл весов: https://disk.yandex.ru/d/7s1i1rJq-LDRIw
Скачать готовые фото для проверки: https://disk.yandex.ru/d/j22X9XsWIcQ1SQ
### Клонирование репозитория
```
git clone https://github.com/LoopLip/Gender-and-age-predict.git
cd Gender-and-age-predict
```

### Установка и активация виртуального окружения
Windows PowerShell:
```
python -m venv venv
venv\Scripts\Activate.ps1
```
Windows CMD:
```
python -m venv venv
venv\Scripts\activate.bat
```

### Установка зависимостей
```
pip install -r requirements.txt
```

Примечание про dlib на Windows:
- Установка dlib через pip может требовать наличия Visual C++ Build Tools и CMake.
- Удобная альтернатива: Anaconda/Miniconda:
```
conda install -c conda-forge dlib
```

## 3. Скачивание данных
### Полный датасет
Рекомендуется использовать IMDb-WIKI или UTK Faces для реального обучения.
- Распакуйте изображения в `data\\imdb_crop` (или в `data\\<db>_crop` при другом db),
- Поместите CSV-метаданные в `meta\\imdb.csv` (или `meta\\<db>.csv`).

Структура:
```
meta\\imdb.csv
data\\imdb_crop\\000001.jpg
data\\imdb_crop\\000002.jpg
...
```

### Мини-датасет (для тестирования)
Если у вас нет полного набора данных, можно сгенерировать мини-датасет для проверки:
```
python run_training.py --generate-demo
```
Это создаст пару простых изображений и CSV в `meta/` и `data/`, чтобы проверить пайплайн.

## 4. Запуск инференса (предсказания)
### Вариант: пакетный инференс
```
python -m src.inference --image_dir data\\imdb_crop --output results.json --batch_size 16 --save_crops
```
Параметры:
- `--image_dir` — папка с изображениями,
- `--output` — куда сохранить результаты (JSON/CSV),
- `--batch_size` — размер батча,
- `--save_crops` — флаг для сохранения кропов лиц.

## 5. Обучение модели
### Быстрый тест (мини-данные)
```
python run_training.py --generate-demo
```
Или напрямую:
```
python train.py --generate-demo
```

### Обучение на полном датасете
1. Положите метаданные в `meta\\<db>.csv` и изображения в `data\\<db>_crop`.
2. Запустите:
```
python run_training.py
```
Опционально можно скачать архив с данными и распаковать:
```
python run_training.py --download-dataset <URL>
```

После каждой эпохи в логах будут: loss/metrics и несколько sample-предсказаний (expected age, top-5 вероятностей).

## 6. Демонстрация (demo)
Для запуска демо (в реальном времени или на видео):
```
python demo.py
```
Предсказания на готовых изображениях, скачанных в test_images
```
python demo.py --image_dir "test_images" --weight_file ".\pretrained_models\EfficientNetB3_224_your_best_weights.hdf5"
```
По умолчанию demo.py ищет последние чекпоинты в папке `checkpoint/`. Чтобы указать конкретный файл весов:
```
python demo.py --weight_file pretrained_models\EfficientNetB3_224_your_best_weights.hdf5
```
Автозагрузка весов: можно указать URL в `src/config.yaml` (параметр `demo.weights_url`) или через переменную окружения `PRETRAINED_WEIGHTS_URL`.

Дополнительные инструкции для переноса и использования готового файла весов (pretrained_models/your_best_weights.hdf5):

1) Создайте папку pretrained_models в корне проекта (если ещё нет):

    mkdir pretrained_models

2) Скачайте файл весов в эту папку (PowerShell): https://disk.yandex.ru/d/7s1i1rJq-LDRIw

3) Запуск demo с явным путем к весам:

    python demo.py --weight_file pretrained_models\EfficientNetB3_224_your_best_weights.hdf5

4) Если предсказания отличаются от оригинального ПК, убедитесь, что demo применяет ту же нормализацию, что и обучающий генератор (в train: img = img.astype(np.float32)/255.0, затем preprocess_fn). В demo при необходимости добавьте перед вызовом preprocess_fn строку:

```python
faces_np = faces_np.astype(np.float32) / 255.0
```

5) Совместимость версий: используйте те же версии TensorFlow/Keras, как и на машине, где модель обучалась, чтобы избежать несовместимостей при загрузке/инференсе.

## 7. Диагностика модели
Быстрая проверка работы модели — вывод expected_age и top-5 вероятностей для N изображений:
```
python tools\\diagnose_model.py --n 3
```
Для корректной работы убедитесь, что есть изображения и CSV в `data/` и `meta/`.

## 8. Нововведения и улучшения
- Генерация мини-датасета для отладки (`--generate-demo`).
- Callback для логирования sample-предсказаний после каждой эпохи.
- `run_training.py` использует текущий интерпретатор Python и умеет скачивать zip-архив датасета.
- `demo.py` автоматически выбирает последний чекпойнт, если вес не указан.
- `tools/diagnose_model.py` — утилита для быстрой диагностики.

## 9. Контакты и дополнительные материалы
- Репозиторий: https://github.com/LoopLip/Gender-and-age-predict
- Датасеты: IMDb-WIKI (поиск в интернете), UTK Faces.

---

Дата обновления: 2025-12-12T22:40:53.397Z
