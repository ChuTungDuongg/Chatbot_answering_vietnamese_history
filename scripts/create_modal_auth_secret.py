"""Create a named Modal authentication Secret without saving a plaintext password."""
import argparse
import getpass
import secrets
import sys

import modal
from argon2 import PasswordHasher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="vn-history-auth", help="New Modal Secret name")
    args = parser.parse_args()
    password = getpass.getpass("Mật khẩu Sử Việt (không hiển thị): ")
    confirm = getpass.getpass("Nhập lại mật khẩu: ")
    if len(password) < 12 or password != confirm:
        print("Mật khẩu phải khớp và dài ít nhất 12 ký tự.", file=sys.stderr)
        return 1
    password_hash = PasswordHasher().hash(password)
    del password, confirm
    modal.Secret.objects.create(args.name, {
        "AUTH_PASSWORD_HASH": password_hash,
        "AUTH_SESSION_SECRET": secrets.token_urlsafe(48),
    })
    print("Đã tạo Modal Secret:", args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
