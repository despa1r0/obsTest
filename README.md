# obsTest

MVP для проекта, где Python-скрипт читает жесты рук и мимику с камеры, подбирает похожий мем из папки `memes`, а затем этот результат можно будет отдавать в OBS Virtual Camera для Zoom, Google Meet и других приложений.

## Что уже работает

- захват видео с веб-камеры через OpenCV;
- распознавание до двух рук через MediaPipe Hand Landmarker;
- базовые жесты:
  - `open_palm`
  - `fist`
  - `peace`
  - `thumbs_up`
  - `point`
- распознавание 5 базовых эмоций лица через MediaPipe Face Landmarker:
  - `happy`
  - `sad`
  - `angry`
  - `surprised`
  - `neutral`
- простая сборка тегов кадра из жестов и эмоции;
- поиск наиболее подходящего мема по тегам из `memes/tags.json` или `memes/tags.example.json`;
- вывод текущих распознанных данных поверх видео в реальном времени.

## Что пока не сделано

- автоматическая подмена видеоисточника в OBS;
- отправка выбранной картинки в OBS scene/source;
- переключение источника в OBS Virtual Camera;
- более умное ранжирование мемов по визуальному сходству, а не только по тегам.

## Запуск

```powershell
.venv\Scripts\python.exe -m app.main
```

Если виртуальное окружение ещё не активировано:

```powershell
.venv\Scripts\Activate.ps1
python -m app.main
```

Для эмоций нужен файл модели `models/face_landmarker.task`.

## Как готовить мемы

1. Положите картинки в папку `memes`.
2. Создайте рядом файл `memes/tags.json`.
3. Опишите для каждой картинки теги жестов и эмоций.

Пример:

```json
{
  "images": [
    {
      "file": "double_peace_cat.jpg",
      "tags": ["peace", "left_peace", "right_peace", "happy", "meme"]
    }
  ]
}
```

## Следующий шаг для OBS

Практичный следующий этап:

1. При совпадении тегов выбирать файл мема.
2. Передавать путь к нему в OBS через `obs-websocket`.
3. Обновлять `Image Source` в отдельной сцене.
4. Включать в Zoom или Meet виртуальную камеру OBS.
