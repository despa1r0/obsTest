# obsTest

MVP-проект, где Python-скрипт распознаёт жесты рук и мимику, выбирает подходящий мем из папки `memes`, показывает его в локальном preview и может отправлять выбранную картинку в OBS через `obs-websocket`.

## Что уже умеет

- читать камеру через OpenCV;
- распознавать до двух рук через MediaPipe Hand Landmarker;
- распознавать 5 эмоций лица через MediaPipe Face Landmarker;
- собирать теги кадра из жестов и эмоции;
- выбирать мем по правилам из `memes/tags.json`;
- поддерживать строгие условия через `required_tags`;
- поддерживать анимации мема через поле `effect`;
- показывать чистый режим и debug-режим по клавише `D`;
- отправлять активный мем в OBS `Image Source`, если включён OBS-режим.

## Жесты и эмоции

Жесты:

- `open_palm`
- `fist`
- `peace`
- `thumbs_up`
- `point`

Эмоции:

- `happy`
- `sad`
- `angry`
- `surprised`
- `neutral`

## Запуск

```powershell
.venv\Scripts\python.exe -m app.main
```

Если окружение ещё не активировано:

```powershell
.venv\Scripts\Activate.ps1
python -m app.main
```

## Формат мемов

Каждый мем описывается в `memes/tags.json`.

Пример:

```json
{
  "images": [
    {
      "file": "surprised.jpg",
      "required_tags": ["surprised"],
      "tags": ["surprised", "wow", "meme"],
      "effect": "fade",
      "priority": 10
    },
    {
      "file": "cinema.png",
      "required_tags": ["neutral", "left_open_palm", "right_open_palm"],
      "tags": ["neutral", "open_palm", "left_open_palm", "right_open_palm", "meme"],
      "effect": "reveal",
      "priority": 30
    }
  ]
}
```

Поля:

- `file`: имя картинки в папке `memes`
- `required_tags`: обязательные теги, без них мем не сработает
- `tags`: дополнительные теги для матчинга
- `effect`: `fade` или `reveal`
- `priority`: приоритет, если несколько мемов подходят сразу

## OBS режим

Скрипт умеет обновлять OBS `Image Source` через `obs-websocket`.

Нужны переменные окружения:

```powershell
$env:OBS_ENABLED="1"
$env:OBS_HOST="localhost"
$env:OBS_PORT="4455"
$env:OBS_PASSWORD="your_password"
$env:OBS_IMAGE_SOURCE_NAME="MemeImage"
$env:OBS_SCENE_NAME="MemeScene"
.venv\Scripts\python.exe -m app.main
```

`OBS_SCENE_NAME` можно не задавать, если сцену переключать не нужно.

## Как настроить OBS

1. В OBS откройте `Tools -> WebSocket Server Settings`.
2. Включите `Enable WebSocket server`.
3. Запомните порт и пароль.
4. Создайте новую сцену, например `MemeScene`.
5. Добавьте в неё `Image Source`, например `MemeImage`.
6. Укажите любую стартовую картинку.
7. Запустите скрипт с переменными окружения для OBS.
8. Запустите `Start Virtual Camera` в OBS.
9. В Zoom, Meet или другом приложении выберите `OBS Virtual Camera`.

## Что ещё дальше

- добавить режим, где Python сам полностью формирует итоговый кадр для OBS;
- сгладить эмоции и жесты по нескольким кадрам;
- добавить отдельные задержки активации для разных мемов;
- добавить управление настройками через файл конфигурации.
