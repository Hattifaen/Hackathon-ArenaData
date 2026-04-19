import os
import re
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps

    HAS_PIL = True
except ImportError:
    raise ImportError("Установите Pillow: pip install Pillow")

try:
    import pytesseract
    from pytesseract import Output

    HAS_TESSERACT = True
except ImportError:
    raise ImportError("Установите pytesseract: pip install pytesseract\n"
                      "Также установите Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")

import numpy as np


@dataclass
class PDnMatch:
    category: str
    pdn_type: str
    value: str
    context: str
    confidence: float = 1.0


@dataclass
class FileResult:
    file_path: str
    pdn_matches: List[PDnMatch] = field(default_factory=list)
    ocr_text: str = ""
    error: Optional[str] = None


class ImagePreprocessor:
    """Предобработка изображений для улучшения OCR"""

    @staticmethod
    def preprocess(image: Image.Image, method: str = 'auto') -> Image.Image:
        """
        Предобработка изображения для улучшения распознавания
        method: 'simple', 'aggressive', 'auto'
        """
        try:
            # Конвертация в grayscale если нужно
            if image.mode != 'L':
                image = image.convert('L')

            if method == 'simple':
                return ImagePreprocessor._simple_preprocess(image)
            elif method == 'aggressive':
                return ImagePreprocessor._aggressive_preprocess(image)
            else:  # auto
                return ImagePreprocessor._auto_preprocess(image)

        except Exception:
            return image

    @staticmethod
    def _simple_preprocess(image: Image.Image) -> Image.Image:
        """Простая предобработка"""
        # Увеличение контраста
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Увеличение резкости
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)

        return image

    @staticmethod
    def _aggressive_preprocess(image: Image.Image) -> Image.Image:
        """Агрессивная предобработка для плохих изображений"""
        # Увеличение размера в 2 раза для лучшего распознавания
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)

        # Конвертация в grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # Удаление шума
        image = image.filter(ImageFilter.MedianFilter(size=3))

        # Увеличение контраста
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        # Увеличение яркости
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)

        # Бинаризация (адаптивный порог)
        img_array = np.array(image)
        threshold = np.mean(img_array)
        img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
        image = Image.fromarray(img_array)

        return image

    @staticmethod
    def _auto_preprocess(image: Image.Image) -> Image.Image:
        """Автоматический выбор метода предобработки"""
        # Оцениваем качество изображения
        if image.mode != 'L':
            img_gray = image.convert('L')
        else:
            img_gray = image

        img_array = np.array(img_gray)

        # Вычисляем контраст (стандартное отклонение)
        contrast = np.std(img_array)

        # Вычисляем уровень шума
        image_blur = img_gray.filter(ImageFilter.GaussianBlur(radius=2))
        noise = np.mean(np.abs(np.array(img_gray).astype(float) - np.array(image_blur).astype(float)))

        # Выбираем метод
        if contrast < 50 or noise > 30:
            # Плохое качество - агрессивная обработка
            return ImagePreprocessor._aggressive_preprocess(image)
        else:
            # Хорошее качество - простая обработка
            return ImagePreprocessor._simple_preprocess(image)


class PDnDetector:
    """Строгий детектор ПДн с валидацией"""

    def __init__(self, context_chars: int = 75):
        self.context_chars = context_chars
        self.patterns = self._compile_patterns()

    def _compile_patterns(self):
        p = {}

        # Паспорт РФ
        p['passport'] = [
            (re.compile(r'паспорт[а-яё\s]*[:\s]*(\d{4}[\s-]?\d{6})\b', re.I), 'Паспорт РФ'),
            (re.compile(r'\b(\d{2}[\s-]\d{2}[\s-]\d{6})\b'), 'Паспорт РФ'),
            (re.compile(r'(?:серия|номер)[\s:]*(\d{4}[\s-]?\d{6})\b', re.I), 'Паспорт РФ'),
            (re.compile(r'\b(\d{10})\b'), 'Паспорт РФ (10 цифр)')
        ]

        # СНИЛС
        p['snils'] = [
            (re.compile(r'\b(\d{3}-\d{3}-\d{3}\s\d{2})\b'), 'СНИЛС'),
            (re.compile(r'СНИЛС[:\s]*(\d{3}-\d{3}-\d{3}\s?\d{2})\b', re.I), 'СНИЛС'),
            (re.compile(r'\b(\d{3}-\d{3}-\d{3}-\d{2})\b'), 'СНИЛС')
        ]

        # ИНН
        p['inn'] = [
            (re.compile(r'\bИНН[:\s]*(\d{10})\b', re.I), 'ИНН'),
            (re.compile(r'\bИНН[:\s]*(\d{12})\b', re.I), 'ИНН'),
            (re.compile(r'\b(\d{12})\b'), 'ИНН (12 цифр)'),
            (re.compile(r'\b(\d{10})\b'), 'ИНН (10 цифр)')
        ]

        # Банковские карты
        p['card'] = [
            (re.compile(r'\b(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{1,5})\b'), 'Банковская карта'),
            (re.compile(r'(?:card|карта)[\s:]*№?[\s:]*(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})\b', re.I),
             'Банковская карта'),
            (re.compile(r'\b(\d{16})\b'), 'Банковская карта (16 цифр)')
        ]

        # Телефоны
        p['phone'] = [
            (re.compile(r'\b(\+7|8)\s*\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{2})[\s.-]?(\d{2})\b'), 'Телефон'),
            (re.compile(r'\bтел\.?[:\s]*(\+7|8)[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b', re.I),
             'Телефон'),
            (re.compile(r'\b(\+7|8)\s*\(?\d{3}\)?\d{7}\b'), 'Телефон')
        ]

        # Email
        p['email'] = [
            (re.compile(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'), 'Email'),
            (re.compile(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,3})\b'), 'Email')
        ]

        # ФИО
        p['fio'] = [
            (re.compile(r'\b([А-ЯЁ][а-яё]+(?:[-\s][А-ЯЁ][а-яё]+)?)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\b'), 'ФИО'),
            (re.compile(r'(?:фио|фамилия|имя|отчество)[:\s]*([А-ЯЁ][а-яё\s-]+)', re.I), 'ФИО'),
            (re.compile(r'\b([А-ЯЁ][а-яё]+(?:[-\s][А-ЯЁ][а-яё]+)?)\b'), 'ФИО (частично)')
        ]

        # Даты рождения
        p['dob'] = [
            (
            re.compile(r'(?:дата\s+рождения|род\.?(?:ился|илась)?)[:\s]*(\d{1,2}[\s./-]\d{1,2}[\s./-]\d{2,4})\b', re.I),
            'Дата рождения'),
            (re.compile(r'\b(\d{2}\.\d{2}\.\d{4})\b'), 'Дата рождения'),
            (re.compile(r'\b(\d{1,2}[\s./-]\d{1,2}[\s./-]\d{2,4})\b'), 'Дата')
        ]

        # Адреса
        p['address'] = [
            (re.compile(r'(?:адрес|прожив\.?(?:ает|ает)|регистрация)[:\s]*([^,\n]{20,150})', re.I), 'Адрес'),
            (re.compile(r'\b(ул\.?\s+[А-Яа-яЁё\s.-]+,\s*(?:д\.?\s*\d+[а-яА-ЯЁё]?)?(?:,\s*кв\.?\s*\d+)?)\b'), 'Адрес'),
            (re.compile(r'\b(г\.?\s*[А-Яа-яЁё]+,\s*(?:ул\.?\s*[А-Яа-яЁё\s.-]+)?)\b'), 'Адрес')
        ]

        # Биометрия и спец. категории
        p['biometric'] = [
            (re.compile(r'\b(?:отпечат[ои]к[и]\s*(?:пальцев|ладони)|дактилоскопическ[ыи]е?\s*данные)\b', re.I),
             'Биометрия'),
            (re.compile(r'\b(?:радужн[ао]я?\s*оболочк[аи]|сканирован[ие]я?\s*сетчатк[иы])\b', re.I), 'Биометрия')
        ]
        p['special'] = [
            (re.compile(r'\b(?:диагноз|заболевани[ея]|медицинск[иы]е?\s*данные|здоровь[ея])\b', re.I),
             'Спец. категория'),
            (re.compile(r'\b(?:религиозн[ыи]е?\s*взгляд[ыы]|вероисповедани[ея])\b', re.I), 'Спец. категория')
        ]

        return p

    def _validate_passport(self, val: str) -> bool:
        d = re.sub(r'[\s-]', '', val)
        if len(d) != 10: return False
        try:
            return 1000 <= int(d[:4]) <= 9999
        except:
            return False

    def _validate_snils(self, val: str) -> bool:
        d = re.sub(r'[\s-]', '', val)
        if len(d) != 11: return False
        try:
            chk = sum(int(d[i]) * (9 - i) for i in range(9)) % 101
            return (0 if chk == 100 else chk) == int(d[9:])
        except:
            return False

    def _validate_inn(self, val: str) -> bool:
        if len(val) not in (10, 12) or not val.isdigit(): return False
        try:
            if len(val) == 10:
                c = [2, 4, 10, 3, 5, 9, 4, 6, 8, 0]
                return (sum(int(val[i]) * c[i] for i in range(9)) % 11) % 10 == int(val[9])
            else:
                c1, c2 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0, 0], [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0]
                k1 = (sum(int(val[i]) * c1[i] for i in range(10)) % 11) % 10
                k2 = (sum(int(val[i]) * c2[i] for i in range(11)) % 11) % 10
                return k1 == int(val[10]) and k2 == int(val[11])
        except:
            return False

    def _luhn(self, num: str) -> bool:
        try:
            digits = [int(d) for d in num if d.isdigit()]
            if not (13 <= len(digits) <= 19): return False
            s = 0
            for i, d in enumerate(reversed(digits)):
                if i % 2: d = d * 2 - 9 if d * 2 > 9 else d * 2
                s += d
            return s % 10 == 0
        except:
            return False

    def _get_context(self, text: str, start: int, end: int) -> str:
        s = max(0, start - self.context_chars)
        e = min(len(text), end + self.context_chars)
        ctx = text[s:e].replace('\n', ' ')
        if s > 0: ctx = "..." + ctx
        if e < len(text): ctx += "..."
        return ctx.strip()

    def find_pdn(self, text: str, file_path: str) -> List[PDnMatch]:
        if not text or len(text.strip()) < 5:
            return []

        matches = []
        seen = set()

        for cat, patterns in self.patterns.items():
            for pat, pdn_type in patterns:
                for m in pat.finditer(text):
                    val = m.group(1) if m.lastindex else m.group(0)

                    # Пропускаем слишком короткие значения
                    if len(val.strip()) < 4:
                        continue

                    h = hashlib.md5(f"{cat}:{val}:{m.start()}".encode()).hexdigest()
                    if h in seen: continue
                    seen.add(h)

                    # Валидация
                    conf = 1.0
                    if cat == 'card' and not self._luhn(val): continue
                    if cat == 'snils' and not self._validate_snils(val): continue
                    if cat == 'inn' and not self._validate_inn(val): continue
                    if cat == 'passport' and not self._validate_passport(val): continue

                    ctx = self._get_context(text, m.start(), m.end())
                    if len(ctx) < 20: continue

                    matches.append(PDnMatch(category=cat, pdn_type=pdn_type, value=val, context=ctx, confidence=conf))
        return matches


class ImageScanner:
    """Улучшенный сканер изображений с множественными попытками OCR"""

    SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.TIF', '.TIFF', '.PNG', '.JPG', '.JPEG'}

    def __init__(self, root_dir: str, threads: int = 12):
        self.root = Path(root_dir)
        self.threads = threads
        self.detector = PDnDetector(context_chars=75)
        self.preprocessor = ImagePreprocessor()
        self.results = []
        self.lock = threading.Lock()
        self.processed = 0
        self.total = 0
        self.stats = {'tif': 0, 'png': 0, 'jpg': 0, 'with_pdn': 0}

    def _collect_files(self) -> List[Path]:
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.root.rglob(f'*{ext}'))
        return list(set(files))

    def _ocr_with_multiple_methods(self, image: Image.Image, lang: str = 'rus+eng') -> str:
        """
        Множественные попытки OCR с разными настройками
        """
        all_text = []

        # Конфигурации Tesseract
        configs = [
            # (PSM, описание)
            (3, 'Fully automatic page segmentation'),
            (6, 'Assume a single uniform block of text'),
            (11, 'Sparse text'),
            (13, 'Raw line'),
        ]

        for psm, desc in configs:
            try:
                custom_config = f'--psm {psm} -c tessedit_char_whitelist=""'
                text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
                if text and text.strip():
                    all_text.append(text.strip())
            except Exception:
                continue

        # Объединяем все найденные тексты
        return "\n".join(all_text)

    def _ocr_image(self, path: Path) -> str:
        """Извлечение текста из изображения с улучшенной обработкой"""
        try:
            img = Image.open(path)
            ext = path.suffix.lower()

            all_texts = []

            # Для многостраничных TIF
            if ext in ('.tif', '.tiff'):
                page = 0
                while True:
                    # Пробуем разные методы предобработки
                    for method in ['auto', 'simple', 'aggressive']:
                        try:
                            preprocessed = self.preprocessor.preprocess(img.copy(), method)
                            text = self._ocr_with_multiple_methods(preprocessed)
                            if text.strip():
                                all_texts.append(f"[Страница {page + 1}, метод {method}]\n{text}")
                        except Exception:
                            pass

                    page += 1
                    try:
                        img.seek(img.tell() + 1)
                    except EOFError:
                        break
            else:
                # PNG, JPG - пробуем разные методы
                for method in ['auto', 'simple', 'aggressive']:
                    try:
                        preprocessed = self.preprocessor.process(img.copy(), method)
                        text = self._ocr_with_multiple_methods(preprocessed)
                        if text.strip():
                            all_texts.append(f"[Метод {method}]\n{text}")
                    except Exception:
                        pass

            return "\n\n".join(all_texts)

        except Exception as e:
            return ""

    def _process(self, path: Path):
        """Обработка одного файла"""
        ext = path.suffix.lower()

        if ext in ('.tif', '.tiff'):
            self.stats['tif'] += 1
        elif ext == '.png':
            self.stats['png'] += 1
        elif ext in ('.jpg', '.jpeg'):
            self.stats['jpg'] += 1

        text = self._ocr_image(path)
        matches = self.detector.find_pdn(text, str(path)) if text.strip() else []

        result = FileResult(file_path=str(path), pdn_matches=matches, ocr_text=text[:500] if text else "")

        with self.lock:
            self.results.append(result)
            self.processed += 1
            if matches:
                self.stats['with_pdn'] += 1

            if self.processed % 5 == 0:
                print(f"Обработано: {self.processed}/{self.total} | "
                      f"С ПДн: {self.stats['with_pdn']} | "
                      f"(TIF: {self.stats['tif']}, PNG: {self.stats['png']}, JPG: {self.stats['jpg']})")

    def scan(self):
        """Запуск сканирования"""
        files = self._collect_files()
        self.total = len(files)

        print(f"🔍 Найдено изображений: {self.total}")
        print(f"   TIF/TIFF: {sum(1 for f in files if f.suffix.lower() in ('.tif', '.tiff'))}")
        print(f"   PNG: {sum(1 for f in files if f.suffix.lower() == '.png')}")
        print(f"   JPG/JPEG: {sum(1 for f in files if f.suffix.lower() in ('.jpg', '.jpeg'))}")
        print()

        if not files:
            print("❌ Изображения не найдены!")
            return []

        print(f"⚙️  Обработка в {self.threads} потоков с улучшенным OCR...")
        print(f"📋 Используемые методы: авто-предобработка, множественные PSM режимы\n")

        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            list(ex.map(self._process, files))

        return self.results


def main():
    share_dir = r"C:\Users\JefTheMax\Desktop\hachaton2026\share"

    if not os.path.isdir(share_dir):
        print(f"❌ Директория не найдена: {share_dir}")
        return

    print("=" * 60)
    print("УЛУЧШЕННОЕ СКАНИРОВАНИЕ ИЗОБРАЖЕНИЙ НА ПДн")
    print("=" * 60)
    print(f"📂 Директория: {share_dir}")
    print(f"📋 Форматы: TIF, PNG, JPG")
    print(f"🔧 Потоков: 12")
    print(f"🎯 OCR: множественные методы + предобработка")
    print("=" * 60 + "\n")

    scanner = ImageScanner(share_dir, threads=12)
    results = scanner.scan()

    pdn_files = [r for r in results if r.pdn_matches]
    pdn_files.sort(key=lambda x: len(x.pdn_matches), reverse=True)

    csv_path = "pdn_images_improved.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for res in pdn_files:
            writer.writerow([Path(res.file_path).name])

    print("\n" + "=" * 60)
    print("✅ СКАНИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"📊 Всего обработано: {scanner.processed}")
    print(f"🎯 Найдено файлов с ПДн: {len(pdn_files)}")
    print(f"📄 Результат: {csv_path}")
    print("=" * 60)

    if pdn_files:
        print("\n🔍 Файлы с персональными данными:")
        for i, res in enumerate(pdn_files[:15], 1):
            pdn_types = {}
            for m in res.pdn_matches:
                pdn_types[m.pdn_type] = pdn_types.get(m.pdn_type, 0) + 1

            types_str = ", ".join([f"{k}: {v}" for k, v in pdn_types.items()])
            print(f"  {i}. {Path(res.file_path).name}")
            print(f"     Найдено: {len(res.pdn_matches)} | {types_str}")

        if len(pdn_files) > 15:
            print(f"  ... и ещё {len(pdn_files) - 15} файлов")
    else:
        print("\n💚 Файлы с ПДн не обнаружены")
        print("\n💡 Рекомендации:")
        print("   1. Проверьте качество изображений")
        print("   2. Убедитесь что Tesseract установлен правильно")
        print("   3. Проверьте наличие русского языка в Tesseract")


if __name__ == "__main__":
    main()