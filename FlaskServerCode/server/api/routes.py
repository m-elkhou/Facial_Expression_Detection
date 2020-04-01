'''
Description:
	Simple server to receive images and responds with results. 
'''

import secrets
import io
from PIL import Image

from datetime import datetime
from flask import Blueprint, request, redirect, url_for, jsonify
import base64
from ..magic_touch import do_magic
import werkzeug
import flask

api = Blueprint('api', __name__, url_prefix='/api')

@api.route('/')
def home():
	return '<h1>Nothing interesting here!</h1>', 404

@api.route('/image', methods=['GET','POST'])
def image():
	try:	
		image_path = f"uploads/"
		imagefile = flask.request.files['image']
		filename = werkzeug.utils.secure_filename(imagefile.filename)
		imagefile.save(image_path+filename)
		
		#image_path = f"uploads/{token}.{ext}"
		#with open(image_path, 'wb') as f:
			#f.write(request.data)
		
		#encodedImg = request.files['pic'] # 'file' is the name of the parameter you used to send the image

		#imgdata = base64.b64decode(encodedImg)

		#filename = 'image.jpg'  # choose a filename. You can send it via the request in an other variable
		
		#with open(image_path, 'wb') as f:
		#	f.write(imgdata)

		return do_magic(image_path+filename)
	except Exception as e:
		return f'<h1>500 Internal Server Error!</h1><br>{e}', 500
	
	
		