from pathlib import Path
from tempfile import TemporaryDirectory
from argon2 import PasswordHasher
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

with TemporaryDirectory(prefix="su-viet-eval-browser-") as directory:
    os.environ.update(
        APP_MODE="api-only",
        APP_ENV="development",
        AUTH_ENABLED="true",
        EVAL_STUDIO_ENABLED="true",
        AUTH_PASSWORD_HASH=PasswordHasher().hash("test-password"),
        AUTH_SESSION_SECRET="test-session-secret-at-least-32-characters",
        CHAT_DATABASE_PATH=str(Path(directory) / "eval.sqlite3"),
        FRONTEND_DIST_DIR=str(Path.cwd() / "frontend" / "dist"),
    )
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8765, log_level="warning")
