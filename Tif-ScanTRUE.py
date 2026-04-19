import os
import re
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Зависимости для OCR
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    raise ImportError("Установите Pillow: pip install Pillow")

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    raise ImportError("Установите pytesseract: pip install pytesseract\n"
                      "Также установите Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")


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
    error: Optional[str] = None


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
            (re.compile(r'(?:серия|номер)[\s:]*(\d{4}[\s-]?\d{6})\b', re.I), 'Паспорт РФ')
        ]

        # СНИЛС
        p['snils'] = [
            (re.compile(r'\b(\d{3}-\d{3}-\d{3}\s\d{2})\b'), 'СНИЛС'),
            (re.compile(r'СНИЛС[:\s]*(\d{3}-\d{3}-\d{3}\s?\d{2})\b', re.I), 'СНИЛС')
        ]

        # ИНН
        p['inn'] = [
            (re.compile(r'\bИНН[:\s]*(\d{10})\b', re.I), 'ИНН'),
            (re.compile(r'\bИНН[:\s]*(\d{12})\b', re.I), 'ИНН'),
            (re.compile(r'\b(\d{12})\b'), 'ИНН')
        ]

        # Банковские карты
        p['card'] = [
            (re.compile(r'\b(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{1,5})\b'), 'Банковская карта'),
            (re.compile(r'(?:card|карта)[\s:]*№?[\s:]*(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})\b', re.I),
             'Банковская карта')
        ]

        # Телефоны
        p['phone'] = [
            (re.compile(r'\b(\+7|8)\s*\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{2})[\s.-]?(\d{2})\b'), 'Телефон'),
            (re.compile(r'\bтел\.?[:\s]*(\+7|8)[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b', re.I),
             'Телефон')
        ]

        # Email
        p['email'] = [
            (re.compile(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'), 'Email')
        ]

        # ФИО
        p['fio'] = [
            (re.compile(r'\b([А-ЯЁ][а-яё]+(?:[-\s][А-ЯЁ][а-яё]+)?)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\b'), 'ФИО'),
            (re.compile(r'(?:фио|фамилия|имя|отчество)[:\s]*([А-ЯЁ][а-яё\s-]+)', re.I), 'ФИО')
        ]

        # Даты рождения
        p['dob'] = [
            (
            re.compile(r'(?:дата\s+рождения|род\.?(?:ился|илась)?)[:\s]*(\d{1,2}[\s./-]\d{1,2}[\s./-]\d{2,4})\b', re.I),
            'Дата рождения'),
            (re.compile(r'\b(\d{2}\.\d{2}\.\d{4})\b'), 'Дата рождения')
        ]

        # Адреса
        p['address'] = [
            (re.compile(r'(?:адрес|прожив\.?(?:ает|ает)|регистрация)[:\s]*([^,\n]{20,150})', re.I), 'Адрес'),
            (re.compile(r'\b(ул\.?\s+[А-Яа-яЁё\s.-]+,\s*(?:д\.?\s*\d+[а-яА-ЯЁё]?)?(?:,\s*кв\.?\s*\d+)?)\b'), 'Адрес')
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
        return len(d) == 10 and 1000 <= int(d[:4]) <= 9999

    def _validate_snils(self, val: str) -> bool:
        d = re.sub(r'[\s-]', '', val)
        if len(d) != 11: return False
        chk = sum(int(d[i]) * (9 - i) for i in range(9)) % 101
        return (0 if chk == 100 else chk) == int(d[9:])

    def _validate_inn(self, val: str) -> bool:
        if len(val) not in (10, 12) or not val.isdigit(): return False
        if len(val) == 10:
            c = [2, 4, 10, 3, 5, 9, 4, 6, 8, 0]
            return (sum(int(val[i]) * c[i] for i in range(9)) % 11) % 10 == int(val[9])
        else:
            c1, c2 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0, 0], [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0]
            k1 = (sum(int(val[i]) * c1[i] for i in range(10)) % 11) % 10
            k2 = (sum(int(val[i]) * c2[i] for i in range(11)) % 11) % 10
            return k1 == int(val[10]) and k2 == int(val[11])

    def _luhn(self, num: str) -> bool:
        digits = [int(d) for d in num if d.isdigit()]
        if not (13 <= len(digits) <= 19): return False
        s = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2: d = d * 2 - 9 if d * 2 > 9 else d * 2
            s += d
        return s % 10 == 0

    def _get_context(self, text: str, start: int, end: int) -> str:
        s = max(0, start - self.context_chars)
        e = min(len(text), end + self.context_chars)
        ctx = text[s:e].replace('\n', ' ')
        if s > 0: ctx = "..." + ctx
        if e < len(text): ctx += "..."
        return ctx.strip()

    def find_pdn(self, text: str, file_path: str) -> List[PDnMatch]:
        matches = []
        seen = set()

        for cat, patterns in self.patterns.items():
            for pat, pdn_type in patterns:
                for m in pat.finditer(text):
                    val = m.group(1) if m.lastindex else m.group(0)
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


class TIFScanner:
    def __init__(self, root_dir: str, threads: int = 12):
        self.root = Path(root_dir)
        self.threads = threads
        self.detector = PDnDetector(context_chars=75)
        self.results = []
        self.lock = threading.Lock()
        self.processed = 0
        self.total = 0

    def _collect_files(self) -> List[Path]:
        exts = ('.tif', '.tiff', '.TIF', '.TIFF')
        files = []
        for ext in exts:
            files.extend(self.root.rglob(f'*{ext}'))
        return list(set(files))

    def _ocr_tif(self, path: Path) -> str:
        try:
            img = Image.open(path)
            texts = []
            while True:
                txt = pytesseract.image_to_string(img, lang='rus+eng').strip()
                if txt: texts.append(txt)
                try:
                    img.seek(img.tell() + 1)
                except EOFError:
                    break
            return "\n".join(texts)
        except Exception:
            return ""

    def _process(self, path: Path):
        text = self._ocr_tif(path)
        matches = self.detector.find_pdn(text, str(path)) if text else []

        with self.lock:
            self.results.append(FileResult(file_path=str(path), pdn_matches=matches))
            self.processed += 1
            if self.processed % 10 == 0:
                print(
                    f"Обработано: {self.processed}/{self.total} | Найдено ПДн: {sum(1 for r in self.results if r.pdn_matches)}")

    def scan(self):
        files = self._collect_files()
        self.total = len(files)
        print(f"Найдено TIF файлов: {self.total}")
        if not files: return []

        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            list(ex.map(self._process, files))
        return self.results


def main():
    share_dir = r"C:\Users\JefTheMax\Desktop\hachaton2026\share"
    if not os.path.isdir(share_dir):
        print(f"Директория не найдена: {share_dir}")
        return

    scanner = TIFScanner(share_dir, threads=12)
    results = scanner.scan()

    # Фильтруем только файлы с ПДн
    pdn_files = [r for r in results if r.pdn_matches]

    # Формируем CSV с одной колонкой name
    csv_path = "pdn_tif_files.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for res in pdn_files:
            # Записываем только имя файла
            writer.writerow([Path(res.file_path).name])

    print(f"\n✅ Готово! Найдено файлов с ПДн: {len(pdn_files)}")
    print(f"📄 Результат сохранён в: {csv_path}")

    if pdn_files:
        print("\n🔍 Примеры найденных файлов:")
        for res in pdn_files[:5]:
            print(f"  • {Path(res.file_path).name} ({len(res.pdn_matches)} совпадений)")


if __name__ == "__main__":
    main()