# obsTest

Черновой MVP для проекта, который в будущем будет подменять видеопоток через OBS на мемы, подобранные по жесту руки или мимике.

## Что уже работает

- захват видео с веб-камеры через OpenCV;
- детекция руки в кадре через MediaPipe Hand Landmarker;
- отрисовка landmarks и связей поверх видео;
- распознавание базовых жестов:
  - `open_palm`
  - `fist`
  - `peace`
  - `thumbs_up`
- вывод названия распознанного жеста в реальном времени;
- подготовленная папка [memes](c:\Users\Vlad%20Solonskyy\Desktop\obsTest\obsTest\memes) для будущих изображений и тегов;
- локальная модель [models/hand_landmarker.task](c:\Users\Vlad%20Solonskyy\Desktop\obsTest\obsTest\models\hand_landmarker.task) для распознавания руки.

## Как запустить

```powershell
.venv\Scripts\python.exe -m app.main
```

Если виртуальное окружение еще не активировано, можно использовать такой вариант:

```powershell
.venv\Scripts\Activate.ps1
python -m app.main
```

## Управление

- `Esc` - мгновенно остановить скрипт
- `Q` - альтернативный выход
- `Ctrl+C` в консоли - аварийная остановка

## Структура проекта

- [app](c:\Users\Vlad%20Solonskyy\Desktop\obsTest\obsTest\app) - основной код приложения
- [memes](c:\Users\Vlad%20Solonskyy\Desktop\obsTest\obsTest\memes) - будущая библиотека мемов
- [models](c:\Users\Vlad%20Solonskyy\Desktop\obsTest\obsTest\models) - локальные модели MediaPipe
- [requirements.txt](c:\Users\Vlad%20Solonskyy\Desktop\obsTest\obsTest\requirements.txt) - зависимости проекта

## Что дальше

Следующий этап разработки:

- загрузка картинок из папки `memes`;
- привязка тегов к изображениям;
- выбор изображения по распознанному жесту;
- вывод результата в OBS.
