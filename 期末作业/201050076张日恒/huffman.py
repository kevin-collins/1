import os, sys
import numpy as np
import cv2
class HuffmanLetter:
    def __init__(self, letter, freq):
        self.letter = letter
        self.freq = freq
        self.bitstring = ""

    def __repr__(self):
        return f"{self.letter}"


class HuffmanTreeNode:
    def __init__(self, freq, left, right):
        self.freq = freq
        self.left = left
        self.right = right


class Huffman:
    def __init__(self):
        self.path = ""
        # self.image_list = [x for x in os.listdir(path) if os.is_image_file(x)]
        # self.image_list = sort\d(self.image_list)

    def byte_cut(self, image):
        image_list = image.flatten()
        chars = {}
        for c in image_list:
            chars[c] = chars[c] + 1 if c in chars.keys() else 1
        return sorted([HuffmanLetter(c, f) for c, f in chars.items()], key=lambda l: l.freq)

    def build_tree(self, letters):
        while len(letters) > 1:
            left = letters.pop(0)
            right = letters.pop(0)
            total_freq = left.freq + right.freq
            node = HuffmanTreeNode(total_freq, left, right)
            letters.append(node)
            letters.sort(key=lambda l: l.freq)

        return letters[0]

    def traverse_tree(self, root, bitstring):
        if type(root) is HuffmanLetter:
            root.bitstring = bitstring
            return [root]
        letters = []
        letters += self.traverse_tree(root.left, bitstring + "0")
        letters += self.traverse_tree(root.right, bitstring + "1")

        return letters

    def test(self):
        test_image = np.array(np.random.randint(0, 25, size=[5, 5]))
        print(test_image.flatten())
        letters_list = self.byte_cut(test_image)
        print(letters_list)
        root = self.build_tree(letters_list)
        letters = self.traverse_tree(root, "")
        dict = {}
        for letter in letters:
            dict[letter.letter] = letter.bitstring

        compress = ""

        for bs in test_image.flatten():
            compress += dict[bs]

        print(sys.getsizeof(test_image.flatten()))
        print(sys.getsizeof(compress))

    def huffman_change(self, image):
        letters_list = self.byte_cut(image)
        root = self.build_tree(letters_list)
        letters = self.traverse_tree(root, "")
        dict = {}
        for letter in letters:
            dict[letter.letter] = letter.bitstring

        compress = ""
        for bs in image.flatten():
            compress += dict[bs]

        return compress, dict

    def compress(self,n):
        outdir = './result4/'
        for index in range(5):
            index= index + 1
            imgpath = 'img' + str(index) + '.jpg'
            image = cv2.imread(imgpath, 1)
            image = cv2.resize(image,(200,200))
            height, width = image.shape[:2]
            if height % n != 0 or width % n != 0:
                image = np.pad(image, ((0, (n - height % n) % n), (0, (n - width % n) % n), (0, 0)),
                               "edge")
            height, width = image.shape[:2]
            size = sys.getsizeof((image.flatten()))

            print("Image {}:".format(index))
            print("Origin Image's Size is {:.2f} KB.".format(size / 1024))


            [b, g, r] = cv2.split(image)
            huff = []
            dict = []
            for img in [b, g, r]:
                # 图像分块
                h_patches = np.vsplit(img, height // n)
                for i in range(height // n):
                    wh_patches = np.hsplit(h_patches[i], width // n)
                    for j in range(width // n):
                        huff_img, huff_dict = self.huffman_change(wh_patches[j])
                        dict.append(huff_dict)
                        huff.append(huff_img)




            r_size = sys.getsizeof(huff)
            r_dict_size = sys.getsizeof(dict)
            # r_size = sys.getsizeof(r)
            # r_dict_size = sys.getsizeof(r_b_dict) + sys.getsizeof(
            #     r_g_dict) + sys.getsizeof(r_r_dict)
            r_size_all = r_size + r_dict_size

            print("After Huffman Encoding Image's Size is  {:.2f} KB.\
                    \nCompressed Image's Huffman coding size is {:.2f} KB.\
                    \nCompressed Image's Huffman coding dictonary size is {:.2f} KB.\
                    \nCompressed Image's size is {:.2%} of Origin Image.\
                    \nblock size is b {:.2f} ."
                  .format(r_size_all / 1024, r_size / 1024, r_dict_size / 1024,
                         r_size_all / size, n ))

            print()

h = Huffman()
h.compress(8)
h.compress(16)
h.compress(32)
