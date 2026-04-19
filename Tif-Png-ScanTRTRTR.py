import os
import re
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
HAS_PIL = True
import pytesseract
from pytesseract import Output
HAS_TESSERACT = True
import numpy as np


@dataclass
class PDnMatch:
    category: str
    pdn_type: str
    value: str
    context: str
    confidence: float = 1.0
    source: str = 'content'  # 'content' или 'filename' — источник обнаружения

@dataclass
class FileResult:
    file_path: str
    pdn_matches: List[PDnMatch] = field(default_factory=list)
    ocr_text: str = ""
    error: Optional[str] = None
    filename_flags: List[str] = field(default_factory=list)  # 🔹 Новые флаги из имени файла


class ImagePreprocessor:
    """Предобработка изображений для улучшения OCR"""

    @staticmethod
    def preprocess(image: Image.Image, method: str = 'auto') -> Image.Image:
        try:
            if image.mode != 'L':
                image = image.convert('L')

            if method == 'simple':
                return ImagePreprocessor._simple_preprocess(image)
            elif method == 'aggressive':
                return ImagePreprocessor._aggressive_preprocess(image)
            else:
                return ImagePreprocessor._auto_preprocess(image)
        except Exception:
            return image

    @staticmethod
    def _simple_preprocess(image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        return image

    @staticmethod
    def _aggressive_preprocess(image: Image.Image) -> Image.Image:
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        if image.mode != 'L':
            image = image.convert('L')
        image = image.filter(ImageFilter.MedianFilter(size=3))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        img_array = np.array(image)
        threshold = np.mean(img_array)
        img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
        image = Image.fromarray(img_array)
        return image

    @staticmethod
    def _auto_preprocess(image: Image.Image) -> Image.Image:
        if image.mode != 'L':
            img_gray = image.convert('L')
        else:
            img_gray = image
        img_array = np.array(img_gray)
        contrast = np.std(img_array)
        image_blur = img_gray.filter(ImageFilter.GaussianBlur(radius=2))
        noise = np.mean(np.abs(np.array(img_gray).astype(float) - np.array(image_blur).astype(float)))
        if contrast < 50 or noise > 30:
            return ImagePreprocessor._aggressive_preprocess(image)
        else:
            return ImagePreprocessor._simple_preprocess(image)


class PDnDetector:
    """Строгий детектор ПДн с валидацией + анализ имён файлов"""

    # 🔹 Ключевые слова в именах файлов, указывающие на ПДн
    FILENAME_KEYWORDS = {
        # Документы удостоверяющие личность
        'passport': ['паспорт', 'паспортные', 'удостоверение', 'ид', 'id', 'identity'],
        'snils': ['снилс', 'пенсионное', 'страховое', 'свидетельство'],
        'inn': ['инн', 'налоговый', 'налогоплательщик', 'tin'],

        # Финансовые данные
        'card': ['карта', 'card', 'банковск', 'счёт', 'счет', 'bank', 'реквизит'],
        'finance': ['зарплат', 'зарплата', 'справка', '2-ндфл', 'ндфл', 'доход'],

        # Контакты
        'phone': ['телефон', 'тел', 'мобильн', 'контакт', 'связь'],
        'email': ['email', 'почта', 'e-mail', 'электронн'],

        # Персональные данные
        'fio': ['фио', 'фамилия', 'имя', 'отчество', 'ф.и.о', 'person', 'личн'],
        'dob': ['рожден', 'дата', 'день', 'месяц', 'год', 'возраст', 'birth'],
        'address': ['адрес', 'прописк', 'регистрац', 'место', 'жительства', 'прожив'],

        # Медицинские и биометрические
        'medical': ['медицин', 'диагноз', 'здоров', 'болезн', 'лечение', 'медкарта'],
        'biometric': ['биометр', 'отпечат', 'дактилоскоп', 'радужк', 'сетчатк', 'скан'],

        # Конфиденциальность
        'confidential': ['конфиденциальн', 'секретн', 'персональн', 'пдн', '152-фз', 'private', 'confidential'],

        # Сканы и копии
        'scan': ['скан', 'копия', 'copy', 'scan', 'фото', 'image', 'picture']
    }

    # 🔹 Паттерны в именах файлов (регулярные выражения)
    FILENAME_PATTERNS = [
        (re.compile(r'(?:passport|паспорт)[_\s-]*(?:scan|copy|скан|копия)?', re.I), 'Паспорт (по имени)'),
        (re.compile(r'(?:snils|снилс|пенсион)[_\s-]*(?:номер|номерок)?', re.I), 'СНИЛС (по имени)'),
        (re.compile(r'(?:inn|инн|налог)[_\s-]*(?:номер)?', re.I), 'ИНН (по имени)'),
        (re.compile(r'\d{4}[_\s-]?\d{6}', re.I), 'Возможно паспорт (10 цифр в имени)'),
        (re.compile(r'\d{3}[_\s-]?\d{3}[_\s-]?\d{3}[_\s-]?\d{2}', re.I), 'Возможно СНИЛС (по имени)'),
        (re.compile(r'(?:personal|персональн|пдн|конфиденциальн)', re.I), 'Конфиденциальный файл'),
        (re.compile(r'(?:client|клиент|customer|заказчик|patient|пациент)', re.I), 'Данные клиента/пациента'),
    ]

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

    # 🔹 НОВЫЙ МЕТОД: Анализ имени файла на признаки ПДн
    def check_filename_pdn(self, file_path: str) -> Tuple[List[PDnMatch], List[str]]:
        """
        Анализирует имя файла на наличие ключевых слов, указывающих на ПДн
        Возвращает: (список матчей, список флагов-подсказок)
        """
        matches = []
        flags = []

        filename = Path(file_path).name.lower()
        filename_no_ext = Path(file_path).stem.lower()

        # 1. Проверка по ключевым словам
        for category, keywords in self.FILENAME_KEYWORDS.items():
            for keyword in keywords:
                if keyword in filename:
                    flags.append(f"ключевое слово: '{keyword}'")
                    # Добавляем маркерный матч с пониженной уверенностью
                    matches.append(PDnMatch(
                        category=category,
                        pdn_type=f"Подозрение по имени файла",
                        value=f"Имя содержит: {keyword}",
                        context=f"Файл: {Path(file_path).name}",
                        confidence=0.4,  # Низкая уверенность — только по имени
                        source='filename'
                    ))
                    break  # Одно совпадение на категорию достаточно

        # 2. Проверка по регулярным паттернам
        for pattern, description in self.FILENAME_PATTERNS:
            if pattern.search(filename_no_ext):
                flags.append(f"паттерн: {description}")
                matches.append(PDnMatch(
                    category='filename_pattern',
                    pdn_type=description,
                    value=f"Совпадение в имени: {Path(file_path).name}",
                    context="Анализ имени файла",
                    confidence=0.5,  # Средняя уверенность для паттернов
                    source='filename'
                ))

        # 3. Проверка на подозрительные форматы имён
        # Например: "Иванов_И.И._паспорт_20240115.tif"
        if re.search(r'[а-яё]+[_\s]+[а-яё]+[_\s]+[а-яё]+', filename_no_ext, re.I):
            # Три слова кириллицей, разделённые _ или пробелом — возможно ФИО
            flags.append("возможно ФИО в имени файла")
            matches.append(PDnMatch(
                category='fio',
                pdn_type="Возможно ФИО (по имени файла)",
                value=Path(file_path).stem,
                context="Структура имени файла",
                confidence=0.3,
                source='filename'
            ))

        return matches, flags

    # === Валидаторы (без изменений) ===
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
        """Поиск ПДн в содержимом + объединение с результатами анализа имени файла"""

        # 🔹 Сначала анализируем имя файла
        filename_matches, filename_flags = self.check_filename_pdn(file_path)

        # Если текст пустой — возвращаем только результаты по имени файла
        if not text or len(text.strip()) < 5:
            return filename_matches

        # Поиск в содержимом (оригинальная логика)
        matches = []
        seen = set()

        for cat, patterns in self.patterns.items():
            for pat, pdn_type in patterns:
                for m in pat.finditer(text):
                    val = m.group(1) if m.lastindex else m.group(0)
                    if len(val.strip()) < 4:
                        continue
                    h = hashlib.md5(f"{cat}:{val}:{m.start()}".encode()).hexdigest()
                    if h in seen: continue
                    seen.add(h)

                    conf = 1.0
                    if cat == 'card' and not self._luhn(val): continue
                    if cat == 'snils' and not self._validate_snils(val): continue
                    if cat == 'inn' and not self._validate_inn(val): continue
                    if cat == 'passport' and not self._validate_passport(val): continue

                    ctx = self._get_context(text, m.start(), m.end())
                    if len(ctx) < 20: continue

                    matches.append(PDnMatch(
                        category=cat, pdn_type=pdn_type, value=val,
                        context=ctx, confidence=conf, source='content'
                    ))

        # 🔹 Объединяем результаты: имя файла + содержимое
        all_matches = filename_matches + matches

        # Дедупликация: если в содержимом найдено то же, что в имени — повышаем уверенность
        for fname_match in filename_matches:
            for content_match in matches:
                if fname_match.category == content_match.category:
                    # Повышаем уверенность контент-матча, если имя файла подтверждает
                    content_match.confidence = min(1.0, content_match.confidence + 0.1)

        # Сортировка по уверенности
        all_matches.sort(key=lambda x: x.confidence, reverse=True)

        return all_matches


class ImageScanner:
    """Сканер TIF-изображений с анализом имён файлов"""

    SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.TIF', '.TIFF'}

    def __init__(self, root_dir: str, threads: int = 12):
        self.root = Path(root_dir)
        self.threads = threads
        self.detector = PDnDetector(context_chars=75)
        self.preprocessor = ImagePreprocessor()
        self.results = []
        self.lock = threading.Lock()
        self.processed = 0
        self.total = 0
        self.stats = {'tif': 0, 'with_pdn': 0, 'by_filename': 0}  # 🔹 Новая статистика

    def _collect_files(self) -> List[Path]:
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.root.rglob(f'*{ext}'))
        return list(set(files))

    def _ocr_with_multiple_methods(self, image: Image.Image, lang: str = 'rus+eng') -> str:
        all_text = []
        configs = [
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
        return "\n".join(all_text)

    def _ocr_image(self, path: Path) -> str:
        try:
            img = Image.open(path)
            all_texts = []
            page = 0
            while True:
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
            return "\n\n".join(all_texts)
        except Exception as e:
            return ""

    def _process(self, path: Path):
        """Обработка одного файла с анализом имени"""
        ext = path.suffix.lower()
        if ext in ('.tif', '.tiff'):
            self.stats['tif'] += 1

        text = self._ocr_image(path)

        # 🔹 Поиск ПДн: теперь включает анализ имени файла
        matches = self.detector.find_pdn(text, str(path)) if text.strip() else []

        # 🔹 Отдельно получаем флаги из имени файла для отчёта
        _, filename_flags = self.detector.check_filename_pdn(str(path))

        result = FileResult(
            file_path=str(path),
            pdn_matches=matches,
            ocr_text=text[:500] if text else "",
            filename_flags=filename_flags  # 🔹 Сохраняем флаги
        )

        with self.lock:
            self.results.append(result)
            self.processed += 1
            if matches:
                self.stats['with_pdn'] += 1
                # 🔹 Считаем файлы, где ПДн найдены хотя бы по имени
                if any(m.source == 'filename' for m in matches):
                    self.stats['by_filename'] += 1

            if self.processed % 5 == 0:
                print(f"Обработано: {self.processed}/{self.total} | "
                      f"С ПДн: {self.stats['with_pdn']} (по имени: {self.stats['by_filename']}) | "
                      f"(TIF: {self.stats['tif']})")

    def scan(self):
        """Запуск сканирования"""
        files = self._collect_files()
        self.total = len(files)

        print(f"🔍 Найдено изображений: {self.total}")
        print(f"   TIF/TIFF: {sum(1 for f in files if f.suffix.lower() in ('.tif', '.tiff'))}")
        print()

        if not files:
            print("❌ TIF-изображения не найдены!")
            return []

        print(f"⚙️  Обработка в {self.threads} потоков...")
        print(f"🔎 Анализ: содержимое + имена файлов на ключевые слова ПДн")
        print()

        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            list(ex.map(self._process, files))

        return self.results


def main():
    share_dir = r"C:\Users\JefTheMax\Desktop\hachaton2026\share"

    if not os.path.isdir(share_dir):
        print(f"❌ Директория не найдена: {share_dir}")
        return

    print("=" * 60)
    print("СКАНИРОВАНИЕ TIF С АНАЛИЗОМ ИМЁН ФАЙЛОВ")
    print("=" * 60)
    print(f"📂 Директория: {share_dir}")
    print(f"📋 Формат: TIF/TIFF")
    print(f"🔍 Детекция: содержимое + ключевые слова в именах")
    print(f"🔧 Потоков: 12")
    print("=" * 60 + "\n")

    scanner = ImageScanner(share_dir, threads=12)
    results = scanner.scan()

    # 🔹 Фильтрация: файлы с ПДн (в содержимом ИЛИ по имени файла)
    pdn_files = [r for r in results if r.pdn_matches]
    pdn_files.sort(key=lambda x: len(x.pdn_matches), reverse=True)

    # 🔹 CSV: ТОЛЬКО ОДНА КОЛОНКА "name"
    csv_path = "pdn_tif_results.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["name"])  # 🔹 Единственная колонка
        for res in pdn_files:
            writer.writerow([Path(res.file_path).name])  # 🔹 Только имя файла

    print("\n" + "=" * 60)
    print("✅ СКАНИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"📊 Всего обработано: {scanner.processed}")
    print(f"🎯 Найдено файлов с ПДн: {len(pdn_files)}")
    print(f"📄 Результат: {csv_path} (колонка: name)")
    print("=" * 60)

    if pdn_files:
        print("\n🔍 Файлы с персональными данными:")
        for i, res in enumerate(pdn_files[:15], 1):
            # Показываем типы найденных ПДн для наглядности в консоли
            pdn_types = {}
            for m in res.pdn_matches:
                pdn_types[m.pdn_type] = pdn_types.get(m.pdn_type, 0) + 1
            types_str = ", ".join([f"{k}: {v}" for k, v in pdn_types.items()])

            # Показываем, если файл попал по имени (для отладки)
            filename_hint = ""
            if any(m.source == 'filename' for m in res.pdn_matches):
                filename_hint = " ⚠️ по имени файла"

            print(f"  {i}. {Path(res.file_path).name}{filename_hint}")
            print(f"     Найдено: {len(res.pdn_matches)} | {types_str}")

        if len(pdn_files) > 15:
            print(f"  ... и ещё {len(pdn_files) - 15} файлов")
    else:
        print("\n💚 Файлы с ПДн не обнаружены")
        print("\n💡 Рекомендации:")
        print("   1. Проверьте качество TIF-изображений")
        print("   2. Убедитесь что Tesseract установлен правильно")
        print("   3. Проверьте наличие русского языка в Tesseract")


if __name__ == "__main__":
    main()