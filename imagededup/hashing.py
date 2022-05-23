#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图片Hashing类和HashEval类."""
import json
import logging
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from scipy.fftpack import dct

from imagededup.hash_search import BruteForceCython

VALID_IMAGE_SUFFIX = {"png", "jpg", "jpeg", "bmp", "gif"}

_LOGGER = logging.getLogger(__name__)


def _check_image_array_hash(image_arr):
    """
    Checks the sanity of the input image numpy array for hashing functions.

    Args:
        image_arr: Image array, np.ndarray.
    """
    image_arr_shape = image_arr.shape
    if len(image_arr_shape) == 3:
        assert image_arr_shape[2] == 3, (f'Received image array with shape: {image_arr_shape}, '
                                         f'expected image array shape is (x, y, 3)')
    elif len(image_arr_shape) > 3 or len(image_arr_shape) < 2:
        raise ValueError(f'Received image array with shape: {image_arr_shape}, expected number of'
                         f' image array dimensions are 3 for rgb image and 2 for grayscale image!')


def _preprocess_image(image, target_size=None, grayscale=False):
    """
    Take as input an image as numpy array or Pillow format. Returns an array version of optionally resized and grayed
    image.

    Args:
        image: numpy array or a pillow image.
        target_size: Size to resize the input image to, Tuple[int, int].
        grayscale: A boolean indicating whether to grayscale the image.

    Returns:
        A numpy array of the processed image.
    """
    if isinstance(image, np.ndarray):
        image = image.astype('uint8')
        image_pil = Image.fromarray(image)

    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise ValueError('Input is expected to be a numpy array or a pillow object!')

    if target_size:
        image_pil = image_pil.resize(target_size, Image.ANTIALIAS)

    if grayscale:
        image_pil = image_pil.convert('L')

    return np.array(image_pil).astype('uint8')


def _load_and_preprocess_image(image_file, target_size=None, grayscale=False, img_formats=None):
    """
    Load an image given its path. Returns an array version of optionally resized and grayed image. Only allows images
    of types described by img_formats argument.

    Args:
        image_file: Path to the image file, Union[PurePath, str]
        target_size: Size to resize the input image to, Tuple[int, int].
        grayscale: A boolean indicating whether to grayscale the image.
        img_formats: List of allowed image formats that can be loaded, List[str].

    Returns:
        A numpy array of the processed image.
    """
    if not img_formats:
        img_formats = VALID_IMAGE_SUFFIX
    try:
        img = Image.open(image_file)

        # validate image format
        if img.format.lower() not in img_formats:
            _LOGGER.warning(f'Invalid image format {img.format}!')
            return None

        else:
            if img.mode != 'RGB':
                # convert to RGBA first to avoid warning
                # we ignore alpha channel if available
                img = img.convert('RGBA').convert('RGB')

            img = _preprocess_image(img, target_size=target_size, grayscale=grayscale)

            return img

    except (IOError, OSError, RuntimeError, ValueError) as e:
        _LOGGER.warning(f'Invalid image file {image_file}:\n{e}')
        return None


def _get_files_to_remove(duplicates):
    """
    Get a list of files to remove.

    Args:
        duplicates: A dictionary with file name as key and a list of duplicate file names as value, Dict[str, List].

    Returns:
        A list of files that should be removed.
    """
    # iterate over dict_ret keys, get value for the key and delete the dict keys that are in the value list
    files_to_remove = set()

    for k, v in duplicates.items():
        tmp = [i[0] if isinstance(i, tuple) else i for i in v]  # handle tuples (image_id, score)

        if k not in files_to_remove:
            files_to_remove.update(tmp)

    return list(files_to_remove)


def _save_json(results, filename, float_scores=False):
    """
    Save results with a filename.

    Args:
        results: Dictionary of results to be saved, Dict.
        filename: Name of the file to be saved.
        float_scores: boolean to indicate if scores are floats.
    """
    _LOGGER.info('Start: Saving duplicates as json!')

    if float_scores:
        for _file, dup_list in results.items():
            if dup_list:
                typecasted_dup_list = []
                for dup in dup_list:
                    typecasted_dup_list.append((dup[0], float(dup[1])))

                results[_file] = typecasted_dup_list

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)

    _LOGGER.info('End: Saving duplicates as json!')


def _parallelise(function, data, verbose):
    """
    Parallelise with verbose.

    Args:
        function: function to be parallelized, Callable.
        data: data to be parallelized, List.
        verbose: show details if set, bool.

    Returns:
        result list.
    """
    pool = Pool(processes=cpu_count())
    results = list(tqdm.tqdm(pool.imap(function, data, 100), total=len(data), disable=not verbose))
    pool.close()
    pool.join()
    return results


class Hashing:
    """
    Find duplicates using hashing algorithms and/or generate hashes given a single image or a directory of images.

    The module can be used for 2 purposes: Encoding generation and duplicate detection.
    - Encoding generation:
    To generate hashes using specific hashing method. The generated hashes can be used at a later time for
    deduplication. Using the method 'encode_image' from the specific hashing method object, the hash for a
    single image can be obtained while the 'encode_images' method can be used to get hashes for all images in a
    directory.

    - Duplicate detection:
    Find duplicates either using the encoding mapping generated previously using 'encode_images' or using a Path to the
    directory that contains the images that need to be deduplicated. 'find_duplicates' and 'find_duplicates_to_remove'
    methods are provided to accomplish these tasks.
    """

    def __init__(self, verbose=True):
        """
        Initialize hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        self.target_size = (8, 8)  # resizing to dims
        self.verbose = verbose

    @staticmethod
    def hamming_distance(hash1, hash2):
        """
        Calculate the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
        to be 64 for each hash and then calculates the hamming distance.

        Args:
            hash1: hash string
            hash2: hash string

        Returns:
            hamming_distance: Hamming distance between the two hashes.
        """
        hash1_bin = bin(int(hash1, 16))[2:].zfill(64)  # zfill ensures that len of hash is 64 and pads MSB if it is < A
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])

    @staticmethod
    def _array_to_hash(hash_mat):
        """
        Convert a matrix of binary numerals to 64 character hash.

        Args:
            hash_mat: A numpy array consisting of 0/1 values.

        Returns:
            An hexadecimal hash string.
        """
        return ''.join('%0.2x' % x for x in np.packbits(hash_mat))

    def encode_image(self, image_file=None, image_array=None):
        """
        Generate hash for a single image.

        Args:
            image_file: Path to the image file.
            image_array: Optional, used instead of image_file. Image typecast to numpy array.

        Returns:
            hash: A 16 character hexadecimal string hash for the image.

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        myhash = myencoder.encode_image(image_file='path/to/image.jpg')
        OR
        myhash = myencoder.encode_image(image_array=<numpy array of image>)
        ```
        """
        try:
            if image_file and os.path.exists(image_file):
                image_file = Path(image_file)
                image_pp = _load_and_preprocess_image(
                    image_file=image_file, target_size=self.target_size, grayscale=True)

            elif isinstance(image_array, np.ndarray):
                _check_image_array_hash(image_array)  # Do sanity checks on array
                image_pp = _preprocess_image(image=image_array, target_size=self.target_size, grayscale=True)
            else:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError('Please provide either image file path or image array!')

        return self._hash_func(image_pp) if isinstance(image_pp, np.ndarray) else None

    def encode_images(self, image_dir=None):
        """
        Generate hashes for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.

        Returns:
            dictionary: A dictionary that contains a mapping of filenames and corresponding 64 character hash string
                        such as {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        mapping = myencoder.encode_images('path/to/directory')
        ```
        """
        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        image_dir = Path(image_dir)

        files = [i.absolute() for i in image_dir.glob('*') if not i.name.startswith('.')]  # ignore hidden files

        _LOGGER.info(f'Start: Calculating hashes...')

        hashes = _parallelise(self.encode_image, files, self.verbose)
        hash_initial_dict = dict(zip([f.name for f in files], hashes))
        # To ignore None (returned if some probelm with image file)
        hash_dict = {k: v for k, v in hash_initial_dict.items() if v}

        _LOGGER.info(f'End: Calculating hashes!')
        return hash_dict

    def _hash_algo(self, image_array):
        """hash algorithm.

        Args:
            image_array: np array of image.
        """
        pass

    def _hash_func(self, image_array):
        """convert image array to hash code.

        Args:
            image_array: np array of image.

        Returns:
            64 character hash.
        """
        hash_mat = self._hash_algo(image_array)
        return self._array_to_hash(hash_mat)

    # search part

    @staticmethod
    def _check_hamming_distance_bounds(thresh):
        """
        Check if provided threshold is valid. Raises TypeError if wrong threshold variable type is passed or a
        ValueError if an out of range value is supplied.

        Args:
            thresh: Threshold value (must be int between 0 and 64)

        Raises:
            TypeError: If wrong variable type is provided.
            ValueError: If invalid value is provided.
        """
        if not isinstance(thresh, int):
            raise TypeError('Threshold must be an int between 0 and 64')
        elif thresh < 0 or thresh > 64:
            raise ValueError('Threshold must be an int between 0 and 64')
        else:
            return None

    def _find_duplicates_dict(self, encoding_map, max_distance_threshold=10, scores=False, outfile=None):
        """
        Take in dictionary {filename: encoded image}, detects duplicates below the given hamming distance threshold
        and returns a dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
        the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            encoding_map: Dictionary with keys as file names and values as encoded images (hashes), Dict[str, str].
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether hamming distance scores are to be returned along with retrieved
            duplicates.
            outfile: Optional, name of the file to save the results. Default is None.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        _LOGGER.info('Start: Evaluating hamming distances for getting duplicates')

        result_set = HashEval(test=encoding_map, queries=encoding_map, distance_function=self.hamming_distance,
                              verbose=self.verbose, threshold=max_distance_threshold)

        _LOGGER.info('End: Evaluating hamming distances for getting duplicates')

        self.results = result_set.retrieve_results(scores=scores)
        if outfile:
            _save_json(self.results, outfile)
        return self.results

    def _find_duplicates_dir(self, image_dir, max_distance_threshold=10, scores=False, outfile=None):
        """
        Take in path of the directory in which duplicates are to be detected below the given hamming distance
        threshold. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            image_dir: Path to the directory containing all the images.
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            outfile: Name of the file the results should be written to.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        encoding_map = self.encode_images(image_dir)
        results = self._find_duplicates_dict(encoding_map=encoding_map, max_distance_threshold=max_distance_threshold,
                                             scores=scores, outfile=outfile)
        return results

    def find_duplicates(self, image_dir=None, encoding_map=None, max_distance_threshold=10, scores=False, outfile=None):
        """
        Find duplicates for each file. Takes in path of the directory or encoding dictionary in which duplicates are to
        be detected. All images with hamming distance less than or equal to the max_distance_threshold are regarded as
        duplicates. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the below the given hamming distance could be returned instead of just duplicate filenames for each
        query file.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            encoding_map: Optional,  used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding hashes.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            scores: Optional, boolean indicating whether Hamming distances are to be returned along with retrieved
                    duplicates.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.

        Returns:
            duplicates dictionary: if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.
                        jpg',score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}. if scores is False, then a
                        dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'],
                        'image2.jpg':['image1_duplicate1.jpg',..], ..}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True,
        outfile='results.json')

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, scores=True, outfile='results.json')
        ```
        """
        self._check_hamming_distance_bounds(thresh=max_distance_threshold)
        if image_dir:
            result = self._find_duplicates_dir(image_dir=image_dir, max_distance_threshold=max_distance_threshold,
                                               scores=scores, outfile=outfile)
        elif encoding_map:
            result = self._find_duplicates_dict(encoding_map=encoding_map,
                                                max_distance_threshold=max_distance_threshold,
                                                scores=scores, outfile=outfile)
        else:
            raise ValueError('Provide either an image directory or encodings!')
        return result

    def find_duplicates_to_remove(self, image_dir=None, encoding_map=None, max_distance_threshold=10, outfile=None):
        """
        Give out a list of image file names to remove based on the hamming distance threshold threshold. Does not
        remove the mentioned files.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            encoding_map: Optional, used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding hashes.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.

        Returns:
            duplicates: List of image file names that are found to be duplicate of me other file in the directory.

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory'),
        max_distance_threshold=15)

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, outfile='results.json')
        ```
        """
        result = self.find_duplicates(image_dir=image_dir, encoding_map=encoding_map,
                                      max_distance_threshold=max_distance_threshold, scores=False)
        files_to_remove = _get_files_to_remove(result)
        if outfile:
            _save_json(files_to_remove, outfile)
        return files_to_remove


class PHash(Hashing):
    """
    Inherits from Hashing base class and implements perceptual hashing (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html).

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Perceptual hash for images
    from imagededup.methods import PHash
    phasher = PHash()
    perceptual_hash = phasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    perceptual_hash = phasher.encode_image(image_array = <numpy image array>)
    OR
    perceptual_hashes = phasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import PHash
    phasher = PHash()
    duplicates = phasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = phasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import PHash
    phasher = PHash()
    files_to_remove = phasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = phasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self, verbose=True):
        """
        Initialize perceptual hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        super().__init__(verbose)
        self.__coefficient_extract = (8, 8)
        self.target_size = (32, 32)

    def _hash_algo(self, image_array):
        """
        Get perceptual hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the perceptual hash of the image.
        """
        dct_coef = dct(dct(image_array, axis=0), axis=1)

        # retain top left 8 by 8 dct coefficients
        dct_reduced_coef = dct_coef[: self.__coefficient_extract[0], : self.__coefficient_extract[1]]

        # median of coefficients excluding the DC term (0th term)
        # mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:])
        median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = dct_reduced_coef >= median_coef_val
        return hash_mat


class HashEval:
    def __init__(self, test, queries, distance_function, verbose=True, threshold=5):
        """
        Initialize a HashEval object which offers an interface to control hashing and search methods for desired
        dataset. Compute a map of duplicate images in the document space given certain input control parameters.

        Args:
            test: test database, encoding map.
            queries: queries, encoding map.
            distance_function: hashing distance function.
            verbose: show details if set. Defaults to True.
            threshold: threshold for distance. Defaults to 5.
        """
        self.test = test  # database
        self.queries = queries
        self.distance_invoker = distance_function
        self.verbose = verbose
        self.threshold = threshold
        self.query_results_map = None
        self._fetch_nearest_neighbors_brute_force_cython()

    def _searcher(self, data_tuple):
        """
        Perform search on a query passed in by _get_query_results multiprocessing part.

        Args:
            data_tuple: Tuple of (query_key, query_val, search_method_object, thresh)

        Returns:
           List of retrieved duplicate files and corresponding hamming distance for the query file.
        """
        query_key, query_val, search_method_object, thresh = data_tuple
        res = search_method_object.search(query=query_val, tol=thresh)
        res = [i for i in res if i[0] != query_key]  # to avoid self retrieval
        return res

    def _get_query_results(self, search_method_object):
        """
        Get result for the query using specified search object. Populate the global query_results_map.

        Args:
            search_method_object: BruteForceCython object to get results for the query.
        """
        args = list(zip(list(self.queries.keys()), list(self.queries.values()),
                        [search_method_object] * len(self.queries), [self.threshold] * len(self.queries)))
        result_map_list = _parallelise(self._searcher, args, self.verbose)
        result_map = dict(zip(list(self.queries.keys()), result_map_list))

        # {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg': [('dup2.jpg', 10)]}
        self.query_results_map = {k: [i for i in sorted(v, key=lambda tup: tup[1], reverse=False)]
                                  for k, v in result_map.items()}

    def _fetch_nearest_neighbors_brute_force_cython(self):
        """
        Wrapper function to retrieve results for all queries in dataset using brute-force search.
        """
        _LOGGER.info('Start: Retrieving duplicates using Cython Brute force algorithm')
        brute_force_cython = BruteForceCython(self.test, self.distance_invoker)
        self._get_query_results(brute_force_cython)
        _LOGGER.info('End: Retrieving duplicates using Cython Brute force algorithm')

    def retrieve_results(self, scores=False):
        """
        Return results with or without scores.

        Args:
            scores: Boolean indicating whether results are to eb returned with or without scores.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        if scores:
            return self.query_results_map
        else:
            return {k: [i[0] for i in v] for k, v in self.query_results_map.items()}
