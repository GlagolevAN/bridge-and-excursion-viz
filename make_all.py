"""Сборка всех статических фигур в out/.

Запуск (из корня репозитория):

    python make_all.py
    python make_all.py fig3 fig9

Каждый модуль figures/fig*.py пишет pdf, png (300 dpi) и svg для пресетов
paper_wide, poster, blog.  Скрипт также проверяет результат (qa.py):
наличие файлов, отсутствие NaN/None в SVG, разумные размеры PNG.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURES = sorted((ROOT / "figures").glob("fig[0-9]*.py"))


def main(argv: list[str]) -> int:
    wanted = [a for a in argv if not a.startswith("-")]
    mods = [f"figures.{p.stem}" for p in FIGURES
            if not wanted or any(w in p.stem for w in wanted)]
    if not mods:
        print("нечего собирать", file=sys.stderr)
        return 2

    t0 = time.time()
    failed = []
    for mod in mods:
        t = time.time()
        proc = subprocess.run([sys.executable, "-m", mod], cwd=ROOT,
                              capture_output=True, text=True)
        status = "ok" if proc.returncode == 0 else "FAIL"
        print(f"[{status}] {mod}  ({time.time() - t:.1f} c)")
        if proc.returncode != 0:
            failed.append(mod)
            print(proc.stderr.strip()[-2000:], file=sys.stderr)

    print(f"\nвсего {len(mods) - len(failed)}/{len(mods)} фигур, "
          f"{time.time() - t0:.1f} c")
    if failed:
        return 1

    qa = subprocess.run([sys.executable, "qa.py"], cwd=ROOT)
    return qa.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
