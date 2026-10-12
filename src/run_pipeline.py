from pathlib import Path
import subprocess
import sys

root_path = Path(__file__).resolve().parent


scripts = ["data/make_dataset.py", "features/build_features.py", "data/validation.py"]


def run_script(script):
    path = root_path / script
    try:

        result = subprocess.run([sys.executable, str(path)], cwd=root_path)
        return result.returncode
    except FileNotFoundError:
        print("Файл скрипта не найден")
        sys.exit(1)


def main():
    for script in scripts:
        code = run_script(script)
        if code != 0:
            print(f"{script} завершился с кодом {code}. Останавливаю работу")
            sys.exit(1)
    print("Pipeline обработки данных выполнен успешно")
    return True


if __name__ == "__main__":
    main()
