from flask import Flask
from .config import Config


def create_app():
	app = Flask(__name__)
	app.config.from_object(Config)
	app.root_path = app.config.get('ROOT_PATH')

	with app.app_context():
		from .api.routes import api
		app.register_blueprint(api)
	return app