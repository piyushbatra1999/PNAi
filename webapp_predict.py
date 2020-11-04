from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import keras.applications.xception as xcep
from PIL import Image

import numpy as np
import time
from datetime import datetime
import pytz
import glob
import os


print('-'*60)

s = time.time()

try:
    model_vgg16 = load_model('Vgg16-crossenropy-softmax-rmsprop_model.h5')
except:
    print('Some error occurred while loading the model!')


print('-'*60)
print(f'Models loaded in: {time.time() - s }')

# just for server info
india_time = pytz.timezone('Asia/Calcutta')
datetime_india = datetime.now(india_time)
machine_time = datetime.now()
print('IST time:', datetime_india.strftime("%d-%b-%y %H:%M"))
print('Machine Time:', machine_time.strftime("%d-%b-%y %H:%M"))


print('-'*60)

pred_no_session = 0
test_dir = "./predict"


def predict_vgg16():
    global pred_no_session
    pred_no_session += 1
    start = time.time()

    test_gen = ImageDataGenerator(rescale=1./255,)  # not augmenting test set

    test_set = test_gen.flow_from_directory(test_dir, shuffle=False,
                                            target_size=(200, 200),
                                            batch_size=1,
                                            class_mode='categorical')
    classes = model_vgg16.predict_generator(test_set)
    # some logging stuff
    print('*' * 60)
    print('predict_vgg16')
    print(classes)
    print('Prediction: Normal') if np.argmax(
        classes) == 0 else print('Prediction: Pneumonia')
    print(f'Predict time: {time.time() - start}')
    print('Prediction number of this session:', pred_no_session)
    print('*'*60)
    files = glob.glob('./predict/uploads/*')
    # deleting uploaded files (privacy?)
    for f in files:
        print(f'Deleting {f}')
        os.remove(f)
    print('*'*60)
    return np.argmax(classes)


# j = 'IM-0115-0001.jpeg'
# k = 'person67_virus_126.jpeg'

# predict_vgg16(j)
# predict_vgg16(k)
