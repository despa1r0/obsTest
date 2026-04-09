from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from app.meme_library import MemeMatch


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


@dataclass(slots=True)
class ObsConfig:
    enabled: bool
    host: str
    port: int
    password: str
    image_source_name: str
    scene_name: str
    opacity_filter_name: str

    @classmethod
    def from_env(cls) -> "ObsConfig":
        return cls(
            enabled=os.getenv("OBS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"},
            host=os.getenv("OBS_HOST", "localhost").strip() or "localhost",
            port=int(os.getenv("OBS_PORT", "4455").strip() or "4455"),
            password=os.getenv("OBS_PASSWORD", ""),
            image_source_name=os.getenv("OBS_IMAGE_SOURCE_NAME", "MemeImage").strip() or "MemeImage",
            scene_name=os.getenv("OBS_SCENE_NAME", "").strip(),
            opacity_filter_name=os.getenv("OBS_OPACITY_FILTER_NAME", "").strip(),
        )


class ObsController:
    def __init__(self, config: ObsConfig) -> None:
        self._config = config
        self._client = None
        self._connected = False
        self._last_sent_path: str | None = None
        self._last_error: str | None = None
        self._scene_item_id: int | None = None
        self._source_visible = False
        self._last_opacity: int | None = None

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def connect(self) -> bool:
        if not self._config.enabled:
            return False

        try:
            import obsws_python as obs

            self._client = obs.ReqClient(
                host=self._config.host,
                port=self._config.port,
                password=self._config.password,
                timeout=3,
            )
            self._connected = True
            self._last_error = None

            if self._config.scene_name:
                self._client.set_current_program_scene(self._config.scene_name)
                self._scene_item_id = self._resolve_scene_item_id()
            return True
        except Exception as exc:
            self._client = None
            self._connected = False
            self._last_error = str(exc)
            return False

    def sync_match(self, match: MemeMatch | None, opacity: float) -> bool:
        if not self._connected or self._client is None:
            return False

        visible = match is not None and opacity > 0.01
        image_changed = False

        if match is not None:
            image_path = str(Path(match.file_path).resolve())
            if image_path != self._last_sent_path:
                try:
                    self._client.set_input_settings(
                        self._config.image_source_name,
                        {"file": image_path},
                        True,
                    )
                    self._last_sent_path = image_path
                    image_changed = True
                except Exception as exc:
                    self._last_error = str(exc)
                    return False

        self._sync_visibility(visible)
        self._sync_opacity(opacity if visible else 0.0)

        if self._config.scene_name:
            try:
                self._client.set_current_program_scene(self._config.scene_name)
            except Exception as exc:
                self._last_error = str(exc)

        self._last_error = None
        return image_changed or visible

    def close(self) -> None:
        if self._client is None:
            return
        disconnect = getattr(self._client, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception:
                pass
        self._client = None
        self._connected = False

    def _resolve_scene_item_id(self) -> int | None:
        if not self._config.scene_name or self._client is None:
            return None
        try:
            result = self._client.get_scene_item_id(
                self._config.scene_name,
                self._config.image_source_name,
            )
            return getattr(result, "scene_item_id", None)
        except Exception:
            return None

    def _sync_visibility(self, visible: bool) -> None:
        if not self._config.scene_name or self._client is None:
            return
        if self._scene_item_id is None:
            self._scene_item_id = self._resolve_scene_item_id()
        if self._scene_item_id is None or visible == self._source_visible:
            return
        try:
            self._client.set_scene_item_enabled(
                self._config.scene_name,
                self._scene_item_id,
                visible,
            )
            self._source_visible = visible
        except Exception as exc:
            self._last_error = str(exc)

    def _sync_opacity(self, opacity: float) -> None:
        if not self._config.opacity_filter_name or self._client is None:
            return

        opacity_percent = max(0, min(100, int(round(opacity * 100))))
        if opacity_percent == self._last_opacity:
            return

        try:
            self._client.set_source_filter_enabled(
                self._config.image_source_name,
                self._config.opacity_filter_name,
                True,
            )
            self._client.set_source_filter_settings(
                self._config.image_source_name,
                self._config.opacity_filter_name,
                {"opacity": opacity_percent},
                True,
            )
            self._last_opacity = opacity_percent
        except Exception as exc:
            self._last_error = str(exc)
