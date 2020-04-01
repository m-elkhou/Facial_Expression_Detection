'''
Powred By: Mohammed Ramouchy (a friend of Maazouz & El-Khoo)
License: MIT license
Website: www.ramouchy.com
Created: 8/12/2019
Description:
	Simple server to receive images and responds with results. 
'''

import os

from server import create_app

app = create_app()

if __name__ == '__main__':
	app.run(host=os.getenv('HOST', '127.0.0.1'), port=os.getenv('PORT', 5000), debug=app.config.get('FLASK_DEBUG'))
else:
	raise ImportError('You can\'t import this file:', __name__)