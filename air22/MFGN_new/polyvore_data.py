from PIL import Image
import json
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import re


class DatasetLoader(Dataset):
    def __init__(self, args, split, task, transform):
        super(DatasetLoader, self).__init__()
        self.split = split
        self.task = task
        self.transform = transform
        self.image_path = os.path.join(args.datadir, 'polyvore_outfits', 'images')
        self.cid2rcid = json.load(open(args.datadir + '/cid2rcid.json', 'r'))
        self.word_to_token = np.load(args.datadir + '/word_to_token.npy', allow_pickle=True).item()

        if args.polyvore_split == 'disjoint':
            self.per_outfit = 16
        else:
            self.per_outfit = 19

        rootdir = os.path.join(args.datadir, 'polyvore_outfits', args.polyvore_split)
        if task == 'fitb':
            data_json = os.path.join(rootdir, 'fill_in_blank_{}_new.json'.format(split))
            self.data = json.load(open(data_json, 'r'))
        else:
            data_json = os.path.join(rootdir, 'compatibility_{}_new.json'.format(split))
            self.data = json.load(open(data_json, 'r'))

    def load_img(self, item_id):
        img_path = os.path.join(self.image_path, '%s.jpg' % item_id)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

    def gen_tokens(self, text):
        unk_id = self.word_to_token['<unk>']
        pad_id = self.word_to_token['<pad>']
        regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
        text = regex.sub(' ', text)
        text = [word.lower() for word in text.split()]
        tokens = [self.word_to_token.get(word, unk_id) for word in text]
        if len(tokens) < 5:
            tokens.extend([pad_id] * (5 - len(tokens)))
        else:
            tokens = tokens[:5]
        return tokens

    def fitb(self, sample):
        imgs1 = np.zeros((self.per_outfit, 3, 96, 96), dtype=np.float32)
        imgs2 = np.zeros((self.per_outfit, 3, 96, 96), dtype=np.float32)
        imgs3 = np.zeros((self.per_outfit, 3, 96, 96), dtype=np.float32)
        imgs4 = np.zeros((self.per_outfit, 3, 96, 96), dtype=np.float32)
        texts1 = np.zeros((self.per_outfit, 5))
        texts2 = np.zeros((self.per_outfit, 5))
        texts3 = np.zeros((self.per_outfit, 5))
        texts4 = np.zeros((self.per_outfit, 5))
        categories1 = np.zeros(self.per_outfit)
        categories2 = np.zeros(self.per_outfit)
        categories3 = np.zeros(self.per_outfit)
        categories4 = np.zeros(self.per_outfit)

        length = len(sample['question'])
        for i in range(length):
            im = sample['question'][i]['im']
            img = self.load_img(im)
            imgs1[i] = img
            imgs2[i] = img
            imgs3[i] = img
            imgs4[i] = img

            text = sample['question'][i]['text']
            text_feature = self.gen_tokens(text)
            texts1[i] = text_feature
            texts2[i] = text_feature
            texts3[i] = text_feature
            texts4[i] = text_feature

            category = self.cid2rcid[sample['question'][i]['fine_category']]
            # category = sample['question'][i]['coarse_category']
            categories1[i] = category
            categories2[i] = category
            categories3[i] = category
            categories4[i] = category

        length += 1
        i += 1
        im1 = sample['answers'][0]['im']
        text1 = sample['answers'][0]['text']
        imgs1[i] = self.load_img(im1)
        texts1[i] = self.gen_tokens(text1)
        categories1[i] = self.cid2rcid[sample['answers'][0]['fine_category']]
        # categories1[i] = sample['answers'][0]['coarse_category']

        im2 = sample['answers'][1]['im']
        text2 = sample['answers'][1]['text']
        imgs2[i] = self.load_img(im2)
        texts2[i] = self.gen_tokens(text2)
        categories2[i] = self.cid2rcid[sample['answers'][1]['fine_category']]
        # categories2[i] = sample['answers'][1]['coarse_category']

        im3 = sample['answers'][2]['im']
        text3 = sample['answers'][2]['text']
        imgs3[i] = self.load_img(im3)
        texts3[i] = self.gen_tokens(text3)
        categories3[i] = self.cid2rcid[sample['answers'][2]['fine_category']]
        # categories3[i] = sample['answers'][2]['coarse_category']

        im4 = sample['answers'][3]['im']
        text4 = sample['answers'][3]['text']
        imgs4[i] = self.load_img(im4)
        texts4[i] = self.gen_tokens(text4)
        categories4[i] = self.cid2rcid[sample['answers'][3]['fine_category']]
        # categories4[i] = sample['answers'][3]['coarse_category']

        answer = sample['label']

        return imgs1, imgs2, imgs3, imgs4, texts1, texts2, texts3, texts4, \
               categories1, categories2, categories3, categories4, length, answer

    def compatibility(self, sample):
        label = torch.tensor(int(sample['label']))

        imgs = np.zeros((self.per_outfit, 3, 96, 96), dtype=np.float32)
        texts = np.zeros((self.per_outfit, 5))
        categories = np.zeros(self.per_outfit)

        length = len(sample['items'])
        for i in range(length):
            im = sample['items'][i]['im']
            text = sample['items'][i]['text']
            imgs[i] = self.load_img(im)
            texts[i] = self.gen_tokens(text)
            categories[i] = self.cid2rcid[sample['items'][i]['fine_category']]
            # categories[i] = sample['items'][i]['coarse_category']
        return imgs, texts, categories, length, label

    def __getitem__(self, index):
        if self.task == 'fitb':
            sample = self.data[index]
            imgs1, imgs2, imgs3, imgs4, texts1, texts2, texts3, texts4, \
            categories1, categories2, categories3, categories4, length, answer = self.fitb(sample)
            return imgs1, imgs2, imgs3, imgs4, texts1, texts2, texts3, texts4, \
                   categories1, categories2, categories3, categories4, length, answer
        else:
            sample = self.data[index]
            imgs, texts, categories, length, label = self.compatibility(sample)
            return imgs, texts, categories, length, label

    def __len__(self):
        return len(self.data)