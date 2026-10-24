"""SQLite by default; --backend json or SERVER_BACKEND=json enables legacy JSON."""

import argparse
import os
from dotenv import load_dotenv
from face_search.config import ROOT

from face_search.api import create_app as _create_app, run


def create_app():
    load_dotenv(ROOT / ".env")
    return _create_app(os.getenv("SERVER_BACKEND", "sqlite"))


def __getattr__(name):
    # ASGI references such as main_server:app initialize through the same factory.
    if name == "app":
        application = create_app()
        globals()[name] = application
        return application
    raise AttributeError(name)


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sqlite", "json"), default=os.getenv("SERVER_BACKEND", "sqlite"))
    run(parser.parse_args().backend)
