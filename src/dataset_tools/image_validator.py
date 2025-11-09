import os
from pathlib import Path
from dataclasses import dataclass

from PIL import Image
from tqdm import tqdm

from src.utils.extensions import normalize_extensions

from .extensions import ImageExtensions


@dataclass
class CleanupResult:
    """Результат очистки поврежденных изображений"""
    total_processed: int
    corrupted_removed: int
    removed_files: list[Path]

    @property
    def success_rate(self) -> float:
        """Процент успешно обработанных файлов"""
        if self.total_processed == 0:
            return 0.0
        return ((self.total_processed - self.corrupted_removed) / self.total_processed) * 100


class ImageValidator:

    def __init__(self, extensions: set[str] | None = None):
        self.extensions = extensions or ImageExtensions.get_extensions()
        self.extensions = normalize_extensions(self.extensions)

    def is_corrupted(self, image_path: str | Path) -> bool:
        """ Проверяет, является ли изображение поврежденным

        :param str | Path image_path: Путь к изображению
        :return: True если изображение валидно, False иначе
        """
        try:
            with Image.open(str(image_path)) as img:
                img.verify()

            return True

        except Exception:
            return False

    def find_image_files(self, dir_path: str | Path) -> list[Path]:
        """ Находит все файлы изображений в директории

        :param str | Path dir_path: Директория для поиска
        :return list[Path]: Список путей к файлам изображений
        """
        return [
            file_path for file_path in Path(dir_path).rglob('*')
            if file_path.is_file() and file_path.suffix.lower() in self.extensions
        ]

    def cleanup_corrupted_images(
        self,
        images_path: str | Path,
        verbose: bool = False,
        dry_run: bool = False
    ) -> dict[str, int]:
        """ Удаляет поврежденные изображения из указанной директории

        :param str | Path images_path: Путь к директории с изображениями
        :param bool verbose: Выводить ли информацию об удаленных файлах
        :param bool dry_run: Если True, только показывает что будет удалено
        :return CleanupResult: Результат очистки
        """
        images_dir = Path(images_path)

        if not images_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {images_dir}")

        image_files = self.find_image_files(images_dir)
        removed_files = []

        if verbose:
            print(f"Найдено {len(image_files)} файлов изображений для проверки")

        for image_path in tqdm(image_files, desc="Проверка изображений"):
            if not self.is_corrupted(image_path):
                if not dry_run:
                    os.remove(image_path)
                removed_files.append(image_path)

                if verbose:
                    action = "Будет удален" if dry_run else "Удален"
                    print(f'{action}: {image_path}')

        result = CleanupResult(
            total_processed=len(image_files),
            corrupted_removed=len(removed_files),
            removed_files=removed_files
        )

        result_dict = {
            "total_processed": result.total_processed,
            "corrupted_removed": result.corrupted_removed,
            "removed_files": result.removed_files
        }

        return result_dict
