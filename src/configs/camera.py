from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Параметры инициализации камеры.

    :param int | str source: Индекс камеры или путь до источника видео
    :param int | None width: Целевая ширина кадра
    :param int | None height: Целевая высота кадра
    :param int | None fps: Целевая частота кадров
    :param bool convert_to_rgb: Конвертировать ли BGR в RGB
    """
    source: int | str = 0
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    convert_to_rgb: bool = True
