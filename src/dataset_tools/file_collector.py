import shutil
from typing import Literal
from pathlib import Path

from tqdm import tqdm

from src.utils.extensions import normalize_extensions

from .extensions import TextExtensions, ImageExtensions


class FileCollector:

    def __init__(self, extensions: Literal["images", "text"] | set[str]):

        if isinstance(extensions, str):
            match extensions:
                case "images":
                    self.extensions = ImageExtensions.get_extensions()
                case "text":
                    self.extensions = TextExtensions.get_extensions()
                case _:
                    raise ValueError(f"Неподдерживаемый тип файлов: {extensions}")
        else:
            self.extensions = extensions

        self.extensions = normalize_extensions(self.extensions)

    def collect_files(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        copy_mode: bool = False,
        progress_bar: bool = False
    ) -> None:
        """Собирает файлы из исходной директории в целевую директорию с заданными расширениями

        :param str | Path source_dir: Исходная директория
        :param str | Path target_dir: Целевая директория
        :param bool copy_mode: Если True - копирует файлы, если False - перемещает
        :param bool progress_bar: Если True - показывает прогресс бар
        """
        source_path = Path(source_dir)
        target_path = Path(target_dir)

        target_path.mkdir(parents=True, exist_ok=True)

        file_paths = [
            file_path for file_path in source_path.rglob('*')
            if file_path.is_file() and file_path.suffix.lower() in self.extensions
        ]

        operation = shutil.copy2 if copy_mode else shutil.move

        for file_path in tqdm(file_paths, desc="Обработка файлов", disable=not progress_bar):
            target_file = target_path / file_path.name

            if target_file.exists():
                print(f'Файл {file_path.name} уже существует в целевой директории!')
                continue

            try:
                operation(file_path, target_file)
            except Exception as e:
                print(f'Ошибка при обработке {file_path.name}: {e}')
