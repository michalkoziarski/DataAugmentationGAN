import gzip
import imageio
import logging
import numpy as np
import pickle
import scipy.ndimage.interpolation
import skimage.color
import skimage.transform
import tarfile
import urllib.request

from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm


DATA_PATH = Path(__file__).parent / 'data'
GENERATED_DATA_PATH = Path(__file__).parent / 'generated_data'
TMP_PATH = Path(__file__).parent / 'tmp'

POSSIBLE_AUGMENTATIONS = ['flip', 'rotation', 'scale', 'translation', 'color', 'gaussian_noise', 'snp_noise']


class AbstractContainer(ABC):
    def __init__(self, partition, batch_size=128, augmentations=(), rotation_range=30,
                 scale_range=1.8, translation_range=0.25, gaussian_noise_std=2,
                 snp_noise_probability=0.001, normalize=True, image_size=None,
                 greyscale=False, n_generated_images=0, generated_data_name_suffix=None):
        assert partition in ['train', 'test']

        if partition == 'test':
            assert len(augmentations) == 0
            assert n_generated_images == 0

        for augmentation in augmentations:
            assert augmentation in POSSIBLE_AUGMENTATIONS

        if image_size is not None:
            assert len(image_size) == 2

        self.partition = partition
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.gaussian_noise_std = gaussian_noise_std
        self.snp_noise_probability = snp_noise_probability
        self.normalize = normalize
        self.image_size = image_size
        self.greyscale = greyscale
        self.n_generated_images = n_generated_images
        self.generated_data_name_suffix = generated_data_name_suffix

        if partition == 'train':
            self.shuffling = True
        elif partition == 'test':
            self.shuffling = False
        else:
            raise NotImplementedError

        self.name = None
        self.n_images = None
        self.images = None
        self.labels = None
        self.position = 0

        self._set_name()
        self._load()

    @abstractmethod
    def _set_name(self):
        pass

    @abstractmethod
    def _download_and_unpack(self):
        pass

    def _unpack(self, **data):
        assert self.name is not None

        _unpack(self.name, **data)

    def _load(self):
        assert self.n_images is None
        assert self.images is None
        assert self.labels is None

        partition_path = DATA_PATH / self.name / self.partition

        if not partition_path.exists():
            self._download_and_unpack()

        self.n_images = 0

        for label_path in partition_path.iterdir():
            for image_path in label_path.iterdir():
                if self.n_images == 0 and self.image_size is None:
                    self.image_size = imageio.imread(str(image_path)).shape[:2]

                self.n_images += 1

        self.n_images += self.n_generated_images

        self.images = np.empty([self.n_images, self.image_size[0], self.image_size[1], 1 if self.greyscale else 3],
                               dtype=np.float32)
        self.labels = np.empty(self.n_images, dtype=np.int64)

        current_index = 0

        for label_path in sorted(partition_path.iterdir()):
            label = int(label_path.stem)

            for image_path in sorted(label_path.iterdir()):
                image = imageio.imread(str(image_path))

                if list(image.shape[:2]) != list(self.image_size):
                    image = skimage.transform.resize(image, self.image_size) * 255.
                else:
                    image = image.astype(np.float32)

                if self.greyscale:
                    image = np.expand_dims(skimage.color.rgb2grey(image / 255.) * 255., 2)
                else:
                    image = skimage.color.grey2rgb(image / 255.) * 255.

                self.images[current_index] = image
                self.labels[current_index] = label

                current_index += 1

        if self.n_generated_images > 0:
            n_generated_images_per_class = self.n_generated_images / len(np.unique(self.labels))
            dataset_name = self.name

            if self.generated_data_name_suffix is not None:
                dataset_name += '_%s' % self.generated_data_name_suffix

            generated_data_path = GENERATED_DATA_PATH / dataset_name

            assert n_generated_images_per_class == int(n_generated_images_per_class)
            assert generated_data_path.exists()

            n_generated_images_per_class = int(n_generated_images_per_class)

            for label_path in sorted(generated_data_path.iterdir()):
                label = int(label_path.stem)

                for i in range(n_generated_images_per_class):
                    image_path = label_path / ('%.5d.png' % (i + 1))
                    image = imageio.imread(str(image_path))

                    if list(image.shape[:2]) != list(self.image_size):
                        image = skimage.transform.resize(image, self.image_size) * 255.
                    else:
                        image = image.astype(np.float32)

                    if self.greyscale:
                        image = np.expand_dims(skimage.color.rgb2grey(image / 255.) * 255., 2)
                    else:
                        image = skimage.color.grey2rgb(image / 255.) * 255.

                    self.images[current_index] = image
                    self.labels[current_index] = label

                    current_index += 1

        if self.shuffling:
            self._shuffle()

    def _shuffle(self):
        indices = list(range(self.n_images))

        np.random.shuffle(indices)

        self.images = self.images[indices]
        self.labels = self.labels[indices]

    def _augment(self, image):
        if 'flip' in self.augmentations:
            image = np.fliplr(image)

        if 'scale' in self.augmentations:
            scale = np.random.rand() * (self.scale_range - 1.0) + 1.0
            rescaled_image = skimage.transform.rescale(image / 255.0, scale=scale, mode='reflect') * 255.0

            if rescaled_image.shape[0] > image.shape[0]:
                x = np.random.randint(rescaled_image.shape[0] - image.shape[0])
            else:
                x = 0

            if rescaled_image.shape[1] > image.shape[1]:
                y = np.random.randint(rescaled_image.shape[1] - image.shape[1])
            else:
                y = 0

            image = rescaled_image[x:(x + image.shape[0]), y:(y + image.shape[1])]

        if 'translation' in self.augmentations:
            shift = [
                np.random.randint(
                    -int(self.images.shape[i] * self.translation_range),
                    int(self.images.shape[i] * self.translation_range) + 1
                ) for i in [1, 2]
            ]

            for i in range(image.shape[2]):
                image[:, :, i] = scipy.ndimage.interpolation.shift(image[:, :, i], shift=shift, mode='reflect')

        if 'rotation' in self.augmentations:
            angle = np.random.randint(-self.rotation_range, self.rotation_range + 1)
            image = skimage.transform.rotate(image / 255.0, angle=angle, mode='reflect') * 255.0

        if 'color' in self.augmentations:
            image = np.flip(image, 2)

        if 'gaussian_noise' in self.augmentations:
            noise = np.random.normal(0.0, self.gaussian_noise_std, image.shape)

            image += noise
            image[image < 0.0] = 0.0
            image[image > 255.0] = 255.0

        if 'snp_noise' in self.augmentations:
            mask = np.random.rand(*image.shape)

            image[mask < self.snp_noise_probability / 2] = 0.0
            image[mask > 1 - self.snp_noise_probability / 2] = 255.0

        return image

    def batch(self):
        batch_images = self.images[self.position:(self.position + self.batch_size)].copy()
        batch_labels = self.labels[self.position:(self.position + self.batch_size)].copy()

        for i in range(len(batch_images)):
            batch_images[i] = self._augment(batch_images[i])

        if self.normalize:
            batch_images /= 255.0
            batch_images -= 0.5
            batch_images *= 2.0

        self.position += self.batch_size

        if self.position >= self.n_images:
            self.position = 0

            if self.shuffling:
                self._shuffle()

        return batch_images, batch_labels

    def batches_per_epoch(self):
        return int(np.ceil(self.n_images / self.batch_size))

    def epoch(self):
        for _ in range(self.batches_per_epoch()):
            yield self.batch()


class MNISTContainer(AbstractContainer):
    def _set_name(self):
        self.name = 'MNIST'

    def _download_and_unpack(self):
        data = {}
        urls = {
            'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        }

        for content in ['train_images', 'train_labels', 'test_images', 'test_labels']:
            file_path = _download(urls[content])

            with gzip.open(file_path, 'rb') as f:
                if content.endswith('images'):
                    data[content] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28, 1)
                    data[content] = np.repeat(data[content], 3, axis=3)
                else:
                    data[content] = np.frombuffer(f.read(), np.uint8, offset=8)

        self._unpack(**data)


class CIFAR10Container(AbstractContainer):
    def _set_name(self):
        self.name = 'CIFAR-10'

    def _download_and_unpack(self):
        data = {}
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

        file_path = _download(url)

        def extract_batch(tar_file, batch_path):
            f = tar_file.extractfile(batch_path)
            d = pickle.loads(f.read(), encoding='bytes')

            images = d[b'data']
            labels = d[b'labels']

            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

            return images, labels

        with tarfile.open(file_path) as f:
            image_batches = []
            label_batches = []

            for i in range(5):
                image_batch, label_batch = extract_batch(f, 'cifar-10-batches-py/data_batch_%d' % (i + 1))
                image_batches.append(image_batch)
                label_batches.append(label_batch)

            data['train_images'] = np.vstack(image_batches)
            data['train_labels'] = np.hstack(label_batches)

            data['test_images'], data['test_labels'] = extract_batch(f, 'cifar-10-batches-py/test_batch')

        self._unpack(**data)


class STL10Container(AbstractContainer):
    def _set_name(self):
        self.name = 'STL-10'

    def _download_and_unpack(self):
        data = {}
        url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

        file_path = _download(url)

        def extract_content(tar_file, content_path):
            f = tar_file.extractfile(content_path)
            content = np.frombuffer(f.read(), dtype=np.uint8)

            return content

        def convert_images(images):
            return images.reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)

        with tarfile.open(file_path) as f:
            data['train_images'] = convert_images(extract_content(f, 'stl10_binary/train_X.bin'))
            data['train_labels'] = extract_content(f, 'stl10_binary/train_y.bin')
            data['test_images'] = convert_images(extract_content(f, 'stl10_binary/test_X.bin'))
            data['test_labels'] = extract_content(f, 'stl10_binary/test_y.bin')

        self._unpack(**data)


def _download(url):
    TMP_PATH.mkdir(parents=True, exist_ok=True)

    file_name = url.split('/')[-1]
    file_path = TMP_PATH / file_name

    logging.info('Downloading %s...' % url)

    with _DownloadProgressBar() as pb:
        urllib.request.urlretrieve(url, file_path, reporthook=pb.update_to)

    return file_path


def _unpack(dataset_name, train_images, train_labels, test_images, test_labels):
    data = {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

    original_classes = sorted(set(data['train_labels']) | set(data['test_labels']))
    n_classes = len(original_classes)
    translated_classes = list(range(n_classes))
    class_translations = {cls: i for cls, i in zip(original_classes, translated_classes)}

    for partition in ['train', 'test']:
        output_directory = DATA_PATH / dataset_name / partition

        logging.info('Saving %s images to %s...' % (partition, output_directory.absolute()))

        class_counts = {}

        for cls in translated_classes:
            (output_directory / str(cls)).mkdir(parents=True, exist_ok=True)

            class_counts[cls] = 0

        images = data['%s_images' % partition]
        labels = data['%s_labels' % partition]

        for image, label in tqdm(zip(images, labels), total=len(images)):
            translated_label = class_translations[label]
            file_name = '%.5d.png' % (class_counts[translated_label] + 1)
            class_counts[translated_label] += 1
            imageio.imwrite(str(output_directory / str(translated_label) / file_name), image)


class _DownloadProgressBar(tqdm):
    def __init__(self, unit='B', unit_scale=True, miniters=1):
        super().__init__(unit=unit, unit_scale=unit_scale, miniters=miniters)

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize

        self.update(b * bsize - self.n)
