# USAGE
# python recognize.py --training images/training --testing images/testing

# import the necessary packages
from py_lbp.localbinarypatterns import LocalBinaryPatterns
from cv2 import cv2
import pickle
import json
from flask import Flask, request, Response
import uuid
import numpy as np


def classify(image):

	desc = LocalBinaryPatterns(8,1)
	
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	#gausian_blur = cv2.GaussianBlur(gray,(5,5),0)
	hist = desc.describe(gray)

	model = pickle.loads(open("model", "rb").read())

	prediction = model.predict(hist.reshape(1, -1))

	# display the image and the prediction
	image = cv2.putText(image, prediction[0], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)

	# save file
	newfile = ('static/%s.jpg' % uuid.uuid4().hex)
	cv2.imwrite(newfile, image)


	return json.dumps(newfile) # return image file name


# API
app = Flask(__name__)

# route http post to this method
@app.route('/api/upload', methods=['POST'])
def upload():
    # retrieve image from client
    img = cv2.imdecode(np.fromstring(request.files['image'].read(),np.uint8),cv2.IMREAD_COLOR)
    # process image
    img_processed = classify(img)
    # response
    return Response (response=img_processed, status=200, mimetype="application/json") # return json string


# start server
if __name__ == "__main__":
    app.run()