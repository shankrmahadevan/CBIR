from PIL import Image
import numpy as np
from pathlib import Path
import time
import ray


@ray.remote
class MultiProcess:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.encoder_models = self.create_model()
        self.path = 'C:\\Users\\mgbal\\Desktop\\Shankar'
        pic_list = []
        for ext in ['jpg', 'png', 'jpeg']:
            pic_list += [str(path) for path in list(Path(self.path).rglob(f'*.{ext}'))]
        self.pic_len = len(pic_list)


    def create_model(self):
        import tensorflow as tf
        models = []
        for i in range(self.num_processes):
            encoder_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')
            encoder_model = tf.keras.models.Model(encoder_model.input, encoder_model.layers[-2].output)
            models.append(encoder_model)
        return models

    def encode_images_mult(self, encoding_train, pic_list, num_process):
        import tensorflow as tf
        mult_fact = self.pic_len // self.num_processes
        for img_path in pic_list[mult_fact * num_process:mult_fact * (num_process + 1)]:
            '''https://docs.google.com/document/d/13ZmoTz5ejiBXMHTi-QYQeDimlXQgwfjQFMayzH2i7dg/edit?usp=sharing'''
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
            img = img.resize((299, 299), Image.ANTIALIAS)
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.inception_v3.preprocess_input(x)
            x = self.encoder_models[num_process].predict(x)
            x = np.reshape(x, 2048)
            encoding_train[img_path] = x
        return encoding_train


def return_encoding_train(path, num_processes=2):
    ray.init(num_cpus=num_processes)
    pic_list = []
    for ext in ['jpg', 'png', 'jpeg']:
        pic_list += [str(path) for path in list(Path(path).rglob(f'*.{ext}'))]
    pic_list_id = ray.put(pic_list)
    start_time = time.time()
    encoding_train = {}
    encoding_train_id = ray.put(encoding_train)
    NetworkActor = MultiProcess.options(max_concurrency=num_processes).remote(num_processes)
    res = ray.get([NetworkActor.encode_images_mult.remote(encoding_train_id, pic_list_id, i) for i in range(num_processes)])
    end_time = time.time()
    total_processing_time = end_time - start_time
    print("Time taken: {}".format(total_processing_time))
    encoding_train = {}
    for dictionary in res:
        encoding_train.update(dictionary)
    return encoding_train
