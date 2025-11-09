from enum import Enum


class ImageExtensions(Enum):
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"

    @classmethod
    def get_extensions(cls) -> set[str]:
        return {ext.value for ext in cls}


class TextExtensions(Enum):
    TXT = ".txt"
    JSON = ".json"

    @classmethod
    def get_extensions(cls) -> set[str]:
        return {ext.value for ext in cls}
