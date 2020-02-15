import os

from flask import jsonify
from model import run
def do_magic(image_path):
	'''Do Your Image mining stuff right here & return some json data to your 
		android app using 'result' variable.
	'''
	# result = {'status': 'done'} # as an example
	result = run(image_path)
	
	# uncommant this line if you want to remove the image after dealing with it.
	#os.remove(image_path)
	return jsonify(result)
