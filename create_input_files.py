import os
from collections import Counter
from random import seed
import numpy as np
import cv2
import h5py
import pandas as pd
from nltk.tokenize import word_tokenize
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from tqdm import tqdm
from random import seed, choice, sample

ROOT_DIR = os.path.abspath(os.curdir)

def create_input_files(captions_per_image, min_word_freq, output_folder,
                       max_len=50):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """
    # word_embedding_model = models.Transformer('bert-base-uncased')
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
    #                                pooling_mode_mean_tokens=True,
    #                                pooling_mode_cls_token=False,
    #                                pooling_mode_max_tokens=False)

    # external_knowledge_embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    # external_knowledge_dic = {}

    # with open(os.path.join(ROOT_DIR,"datasets/external_knowledge.txt")) as fp:
    #     Lines = fp.readlines()
    #     for line in Lines:
    #         image_name, external_knowledge = line.split(';', 1)
    #         enc_external_knowldge = external_knowledge_embedding_model.encode([external_knowledge.strip()])
    #         external_knowledge_dic[image_name] = enc_external_knowldge[0].tolist()

    # print('Finished encoding external knowledge.')
    # Read data
    train_data = pd.read_csv(os.path.join(ROOT_DIR,"datasets/train_data.csv"), sep=';')
    val_data = pd.read_csv(os.path.join(ROOT_DIR,"datasets/val_data.csv"), sep=';')
    test_data = pd.read_csv(os.path.join(ROOT_DIR,"datasets/test_data.csv"), sep=';')

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_knowledge = []
    train_image_opinion = []
    train_image_gender = []
    # train_image_exteral_knowledge = []

    val_image_paths = []
    val_image_captions = []
    val_image_knowledge = []
    val_image_opinion = []
    val_image_gender = []
    # val_image_exteral_knowledge = []

    test_image_paths = []
    test_image_captions = []
    test_image_knowledge = []
    test_image_opinion = []
    test_image_gender = []
    # test_image_exteral_knowledge = []

    word_freq = Counter()

    for _, row in train_data.iterrows():
        path = os.path.join(ROOT_DIR,"datasets/images/{}".format(row['image_path']))
        image_name = row['image_path'].split('-')[0]
        raw_caption = row['caption']
        knowledge = row['knowledge_level']
        opinion = row['opinion']
        gender = row['gender']
        # raw_external_knowledge = row['external_knowledge'].strip()
        # tokenized_external_knowledge = word_tokenize(raw_external_knowledge)
        # enc_external_knowledge = external_knowledge_dic[image_name]
        tokenized_caption = word_tokenize(raw_caption)
        # word_freq.update(tokenized_external_knowledge)
        word_freq.update(tokenized_caption)

        train_image_paths.append(path)
        train_image_captions.append(tokenized_caption)
        train_image_knowledge.append(knowledge)
        train_image_opinion.append(opinion)
        train_image_gender.append(gender)
        # train_image_exteral_knowledge.append(enc_external_knowledge)

    for _, row in val_data.iterrows():
        path = os.path.join(ROOT_DIR,"datasets/images/{}".format(row['image_path']))
        image_name = row['image_path'].split('-')[0]
        raw_caption = row['caption']
        knowledge = row['knowledge_level']
        opinion = row['opinion']
        gender = row['gender']

        # raw_external_knowledge = row['external_knowledge']
        # tokenized_external_knowledge = word_tokenize(raw_external_knowledge)
        # enc_external_knowledge = external_knowledge_dic[image_name]
        tokenized_caption = word_tokenize(raw_caption)
        # word_freq.update(tokenized_external_knowledge)
        word_freq.update(tokenized_caption)
        val_image_paths.append(path)
        val_image_captions.append(tokenized_caption)
        val_image_knowledge.append(knowledge)
        val_image_opinion.append(opinion)
        val_image_gender.append(gender)
        # val_image_exteral_knowledge.append(enc_external_knowledge)

    for _, row in test_data.iterrows():
        path = os.path.join(ROOT_DIR,"datasets/images/{}".format(row['image_path']))
        image_name = row['image_path'].split('-')[0]
        raw_caption = row['caption']
        knowledge = row['knowledge_level']
        opinion = row['opinion']
        gender = row['gender']

        # raw_external_knowledge = row['external_knowledge']
        # tokenized_external_knowledge = word_tokenize(raw_external_knowledge)
        # enc_external_knowledge = external_knowledge_dic[image_name]
        tokenized_caption = word_tokenize(raw_caption)
        # word_freq.update(tokenized_external_knowledge)
        word_freq.update(tokenized_caption)
        test_image_paths.append(path)
        test_image_captions.append(tokenized_caption)
        test_image_knowledge.append(knowledge)
        test_image_opinion.append(opinion)
        test_image_gender.append(gender)
        # test_image_exteral_knowledge.append(enc_external_knowledge)


    # Sanity check
    # print(len(train_image_paths),len(train_image_captions) , len(train_image_knowledge) , len(train_image_opinion) , len(train_image_gender),len(train_image_exteral_knowledge))
    # assert len(train_image_paths) == len(train_image_captions) == len(train_image_knowledge) == len(train_image_opinion) == len(train_image_gender) == len(train_image_exteral_knowledge)
    # assert len(val_image_paths) == len(val_image_captions) == len(val_image_knowledge) == len(val_image_opinion) == len(val_image_gender) ==len(val_image_exteral_knowledge)
    # assert len(test_image_paths) == len(test_image_captions) == len(test_image_knowledge) == len(test_image_opinion) == len(test_image_gender)==len(test_image_exteral_knowledge)

    # Create word map

    # # Read Karpathy JSON

    # karpathy_json_path = '../coco_dataset/karpathy_json/dataset_coco.json'
    # with open(karpathy_json_path, 'r') as j:
    #     data = json.load(j)

    # for img in data['images']:
    #     captions = []
    #     for c in img['sentences']:
    #         # Update word frequency
    #         word_freq.update(c['tokens'])
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_KNOWLEDGE5K.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, knowledge, opinion, gender, split in [(train_image_paths, train_image_captions,
                                                                train_image_knowledge,train_image_opinion, train_image_gender, 'TRAIN'),
                                   (val_image_paths, val_image_captions,
                                    val_image_knowledge, val_image_opinion, val_image_gender ,'VAL'),
                                   (test_image_paths, test_image_captions,
                                    test_image_knowledge, test_image_opinion, test_image_gender,'TEST')]:
        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Read images
                img = cv2.imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = cv2.resize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img


                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in imcaps[i]] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(imcaps[i]))

                # Find caption lengths
                c_len = len(imcaps[i]) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)

            # Sanity check
            # assert images.shape[0] == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_KNOWLEDGE_' + base_filename + '.json'), 'w') as j:
                json.dump(knowledge, j)

            with open(os.path.join(output_folder, split + '_OPINION_' + base_filename + '.json'), 'w') as j:
                json.dump(opinion, j)

            with open(os.path.join(output_folder, split + '_GENDER_' + base_filename + '.json'), 'w') as j:
                json.dump(gender, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            # with open(os.path.join(output_folder, split + '_EXTERNAL_KNOWLEDGE_' + base_filename + '.json'), 'w') as j:
            #     json.dump(external_konwledge, j)

if __name__ == '__main__':
    output_folder=os.path.join(os.path.abspath(os.curdir),'datasets')
    captions_per_image = 1
    min_word_freq=1
    max_len=50
    create_input_files(captions_per_image, min_word_freq, output_folder, max_len=50):
