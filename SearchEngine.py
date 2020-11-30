import os
import cv2
from glob import glob
import tensorflow as tf
from tqdm import tqdm
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import wget
from pathlib import Path
from CBIR.multiprocess import return_encoding_train

class SearchEngine:
    def __init__(self):
        encoder_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')
        self.encoder_model = tf.keras.models.Model(encoder_model.input, encoder_model.layers[-2].output)
        self.img_dim = (299, 299)
        self.output_dim = 2048
        self.preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        self.pickle_file = ''
        self.search_dir = ''

    def load_pickle(self, path):
        try:
            self.create_pickle(path)
            with open(path + '/train.pkl', "rb") as fp:
                pickle_file = pickle.load(fp)
            return pickle_file
        except:
           print('Could not load pickle file')
           import sys
           print(sys.exc_info())
           return False

    def create_pickle(self, path):
        train_path = path + '\\train.pkl'
        pic_list = []
        for ext in ['jpg', 'png', 'jpeg']:
            pic_list += [str(path) for path in list(Path(path).rglob(f'*.{ext}'))]
        if os.path.exists(train_path):
            updated = 0
            with open(train_path, "rb") as fp:
                encoding_train = pickle.load(fp)
            keys = [str(path) for path in list(encoding_train.keys())]
            for img_path in tqdm(pic_list):
                if img_path not in keys:
                    updated += 1
                    img = tf.keras.preprocessing.image.load_img(img_path,
                                                                target_size=(299, 299))
                    encoding_train[img_path] = self.encode_image(img)
            deleted = 0
            for key in keys:
                if key not in pic_list:
                    try:
                        encoding_train.pop(key)
                        deleted += 1
                    except:
                        pass
            with open(train_path, "wb") as fp:
                pickle.dump(encoding_train, fp)
            print(f'Found {updated} new images. Deleted {deleted}. {len(encoding_train.keys())} images present now. (Earlier {len(keys)})')
        else:
            encoding_train = return_encoding_train(path)
            with open(train_path, "wb") as fp:
                pickle.dump(encoding_train, fp)
        return os.path.exists(train_path)

    def chi2_distance(self, image1, image2, eps=1e-10):
        distance = 0.5 * np.sum((image1 - image2) ** 2 / (image1 + image2 + eps))
        return distance

    def encode_image(self, img):
        img = img.resize(self.img_dim, Image.ANTIALIAS)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        x = self.encoder_model.predict(x)
        x = np.reshape(x, self.output_dim)
        return x

    def search(self):
        search_dir = input('Input the directory to be searched\n')
        if not self.pickle_file:
            self.pickle_file = self.load_pickle(search_dir)
        self.search_dir = search_dir
        if not os.path.exists('img_folder'):
            os.mkdir('img_folder')
        else:
            for image in glob('img_folder/**'):
                os.remove(image)
        try:
            wget.download(input("Enter The Download Link !\n"), out='img_folder/')
            img_input_path = glob('img_folder/**')[0]
        except:
            print('Could not download image, try some other link')
            return
        test_image = tf.keras.preprocessing.image.load_img(img_input_path,
                                                           target_size=self.img_dim)
        plt.imshow(cv2.imread(img_input_path)[..., ::-1])
        plt.title('Input Image')
        plt.show()
        features = self.encode_image(test_image)
        keys = list(self.pickle_file.keys())
        array = []
        for image in tqdm(keys):
            array.append((image, self.chi2_distance(features, self.pickle_file[image])))
        array.sort(key=lambda x: x[1])
        fig = plt.figure(figsize=(20, 20))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)
        for i in range(10):
            fig.add_subplot(5, 2, i + 1)
            plt.axis('off')
            plt.imshow(cv2.resize(cv2.imread(str(array[i][0]))[..., ::-1], (331, 331)))
        plt.tight_layout()
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1.3)
        fig.subplots_adjust(right=0.9)
        fig.subplots_adjust(left=0)
        plt.show()
        os.remove(img_input_path)


search_engine = SearchEngine()
search_engine.search()
