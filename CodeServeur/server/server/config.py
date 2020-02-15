import os

class Config():
	#FLASK_DEBUG = True if os.environ.get('FLASK_DEBUG') == 1 else False
	#SECRET_KEY = os.environ.get('SECRET_KEY')
	ROOT_PATH = os.path.join(os.getcwd(), __package__)