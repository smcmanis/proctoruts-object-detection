import object_detection_api
from object_detection_api import get_objects

import os
from PIL import Image
from flask import Flask, request, Response

myapp = Flask(__name__)

# for CORS
@myapp.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


@myapp.route('/')
def index():
    return Response('Tensor Flow object detection')


@myapp.route('/local')
def local():
    return Response(open('./static/local.html').read(), mimetype="text/html")


@myapp.route('/video')
def remote():
    return Response(open('./static/video.html').read(), mimetype="text/html")


@myapp.route('/test')
def test():
    PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'  # cwh
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

    image = Image.open(TEST_IMAGE_PATHS[0])
    objects = get_objects(image)

    return objects


@myapp.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image']  # get the image

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        if threshold is None:
            threshold = 0.5
        else:
            threshold = float(threshold)

        # finally run the image through tensor flow object detection`
        image_object = Image.open(image_file)
        objects = get_objects(image_object, threshold)
        return objects

    except Exception as e:
        print('POST /image error: %e' % e)
        return e



	# without SSL
    # myapp.run(debug=True, host='0.0.0.0')

	# with SSL
    #myapp.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))

