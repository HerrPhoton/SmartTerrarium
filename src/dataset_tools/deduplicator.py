from pathlib import Path

from imagededup.utils import plot_duplicates
from imagededup.methods import CNN


class ImageDeduplicator:

    def __init__(self, verbose: bool = False):
        self.cnn = CNN(verbose)
        self.image_dir: str | None = None
        self.duplicates_map: dict[str, list[str]] | None = None
        self.files_with_duplicates: set[str] | None = None
        self.duplicates: set[str] | None = None

    def find_duplicates(
        self,
        image_dir: str | Path,
        min_similarity_threshold: float = 0.9,
    ) -> dict[str, list[str]]:
        """Находит дубликаты изображений в указанной директории.

        :param str | Path image_dir: Путь к директории с изображениями
        :param float min_similarity_threshold: Минимальный порог схожести (0.0-1.0)
        :return dict[str, list[str]]: Словарь, где ключ - имя файла, значения - список имен файлов дубликатов
        """
        self.image_dir = str(image_dir)

        self.duplicates_map = self.cnn.find_duplicates(
            image_dir=image_dir,
            min_similarity_threshold=min_similarity_threshold,
        )

        self.files_with_duplicates = set()
        self.duplicates = set()

        for file, duplicates in self.duplicates_map.items():
            if len(duplicates) > 0:
                self.files_with_duplicates.update(file)
                self.duplicates.update(duplicates)

        return self.duplicates_map

    def delete_duplicates(
        self,
        image_dir: str | Path,
        labels_dir: str | Path | None = None,
        min_similarity_threshold: float = 0.9,
        dry_run: bool = False
    ) -> list[str]:
        """Удаляет дубликаты изображений по заданному порогу.

        :param str image_dir: Путь к директории с изображениями
        :param str | None labels_dir: Путь к директории с метками
        :param float min_similarity_threshold: Минимальный порог схожести (0.0-1.0)
        :param str dry_run: Если True, только показывает что будет удалено, без фактического удаления
        :return list[str]: Список путей к удаленным файлам
        """
        if self.duplicates_map is None:
            self.find_duplicates(image_dir, min_similarity_threshold)

        image_dir = Path(image_dir)
        labels_dir = Path(labels_dir) if labels_dir is not None else None

        removed_files = []
        for file in self.duplicates:
            image_path = image_dir / file
            removed_files.append(str(image_path))

            if not dry_run:
                image_path.unlink()

            if labels_dir is not None:
                label_path = labels_dir / Path(file).with_suffix('.txt')
                removed_files.append(str(label_path))

                if not dry_run:
                    label_path.unlink()

        if not dry_run:
            self.duplicates_map = None
            self.duplicates = None
            self.image_dir = None

        return removed_files

    def visualize_duplicates(
        self,
        target: str | Path | int,
    ) -> None:
        """Визуализирует дубликаты для указанного изображения.

        :param str | Path | int target: Имя файла или индекс для визуализации
        """
        if self.duplicates_map is None:
            raise ValueError(
                "Не найден словарь дубликатов. Сначала вызовите find_duplicates()"
            )

        if isinstance(target, int):
            filename = self.files_with_duplicates[target]
        else:
            filename = str(filename)

        plot_duplicates(
            image_dir=self.image_dir,
            duplicate_map=self.duplicates_map,
            filename=filename,
        )
