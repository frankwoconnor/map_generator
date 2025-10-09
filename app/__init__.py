from flask import Flask

from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Register blueprints
    from app.routes import api, views

    app.register_blueprint(api.bp)
    app.register_blueprint(views.bp)

    return app
