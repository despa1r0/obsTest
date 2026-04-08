from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from app.meme_library import MemeMatch


@dataclass(slots=True)
class ObsConfig:
    enabled: bool
    host: str
    port: int
    password: str
    image_source_name: str
    scene_name: str

    @classmethod
    def from_env(cls) -> "ObsConfig":
        return cls(
            enabled=os.getenv("OBS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"},
            host=os.getenv("OBS_HOST", "localhost").strip() or "localhost",
            port=int(os.getenv("OBS_PORT", "4455").strip() or "4455"),
            password=os.getenv("OBS_PASSWORD", ""),
            image_source_name=os.getenv("OBS_IMAGE_SOURCE_NAME", "MemeImage").strip() or "MemeImage",
            scene_name=os.getenv("OBS_SCENE_NAME", "").strip(),
        )


class ObsController:
    def __init__(self, config: ObsConfig) -> None:
        self._config = config
        self._client = None
        self._connected = False
        self._last_sent_path: str | None = None
        self._last_error: str | None = None

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
            return True
        except Exception as exc:
            self._client = None
            self._connected = False
            self._last_error = str(exc)
            return False

    def sync_match(self, match: MemeMatch | None) -> bool:
        if not self._connected or self._client is None or match is None:
            return False

        image_path = str(Path(match.file_path).resolve())
        if image_path == self._last_sent_path:
            return False

        try:
            self._client.set_input_settings(
                self._config.image_source_name,
                {"file": image_path},
                True,
            )
            if self._config.scene_name:
                self._client.set_current_program_scene(self._config.scene_name)
            self._last_sent_path = image_path
            self._last_error = None
            return True
        except Exception as exc:
            self._last_error = str(exc)
            return False

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
