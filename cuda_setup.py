"""
ctranslate2(faster-whisper)が pip の nvidia-*-cu12 wheel に含まれる
CUDA DLL(cublas64_12.dll / cudnn*.dll 等)を見つけられるようにする。

Windows では DLL 探索ディレクトリを明示登録しないと RuntimeError になるため、
faster_whisper を import する前に本モジュールを import すること。
"""
import os
import site
import glob


def setup():
    bases = set(site.getsitepackages())
    try:
        bases.add(site.getusersitepackages())
    except Exception:
        pass
    added = []
    for base in bases:
        for bindir in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            if os.path.isdir(bindir):
                try:
                    os.add_dll_directory(bindir)
                    added.append(bindir)
                except Exception:
                    pass
    return added


setup()
