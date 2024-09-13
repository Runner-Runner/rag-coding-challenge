import os
from venv import create
from os.path import abspath
import subprocess
import platform


def run():
    venv_name = 'venv_cc'
    venv_path = os.path.join('..', venv_name)
    if os.path.exists(venv_path):
        print("venv already exists.")
        return

    create(venv_path, with_pip=True)

    system = platform.system()  # venv structure is OS-specific
    if system == 'Windows':
        pip_path = os.path.join('Scripts', 'pip')
    else:
        pip_path = os.path.join('bin', 'pip')
    subprocess.run([pip_path, 'install', '-r', abspath(os.path.join('..', 'requirements.txt'))], cwd=venv_path)


if __name__ == '__main__':
    run()
