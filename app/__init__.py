import os
import secrets
from app.config.settings import MAX_CONTENT_LENGTH


def create_app():
    from flask import Flask

    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')

    app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    from app.api import register_blueprints

    register_blueprints(app)

    return app
