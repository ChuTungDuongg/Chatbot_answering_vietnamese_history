"""Modal imports the defining module inside the lightweight CPU container."""
import subprocess
import sys


def test_remote_module_import_does_not_require_local_dotenv():
    result = subprocess.run([sys.executable, "-c", """
import builtins
import modal
modal.is_local = lambda: False
original = builtins.__import__
def import_without_dotenv(name, *args, **kwargs):
    if name == 'dotenv':
        raise ModuleNotFoundError('CPU runtime has no local dotenv dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = import_without_dotenv
import modal_app
print('CPU remote module import: PASS')
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CPU remote module import: PASS" in result.stdout
