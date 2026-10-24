"""SQLite ASGI API. Start with Python or uvicorn main_server_sql:create_app --factory."""

from face_search.api import create_app as _create_app, run


def create_app():
    return _create_app("sqlite")


def __getattr__(name):
    if name == "app":
        application = create_app()
        globals()[name] = application
        return application
    raise AttributeError(name)


if __name__ == "__main__":
    run("sqlite")
