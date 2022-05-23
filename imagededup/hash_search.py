#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""哈希距离搜索算法."""
import brute_force_cython_ext


class BruteForceCython:
    """
    Class to perform search using a Brute force.
    """

    def __init__(self, hash_dict, distance_function):
        """
        Initialize a dictionary for mapping file names and corresponding hashes and a distance function to be used for
        getting distance between two hash strings.

        Args:
            hash_dict: Dictionary mapping file names to corresponding hash strings {filename: hash}
            distance_function:  A function for calculating distance between the hashes.
        """
        self.distance_function = distance_function
        self.hash_dict = hash_dict  # database

    def search(self, query, tol=10):
        """
        Function for searching using brute force.

        Args:
            query: hash string for which brute force needs to work.
            tol: distance upto which duplicate is valid.

        Returns:
            List of tuples of the form [(valid_retrieval_filename1: distance), (valid_retrieval_filename2: distance)].
        """

        filenames = []
        hash_vals = []
        for filename, hash_val in self.hash_dict.items():
            filenames.append(filename.encode('utf-8'))
            hash_vals.append(int(hash_val, 16))

        # cast hex hash_val to decimals for __builtin_popcountll function
        return brute_force_cython_ext.query(filenames, hash_vals, int(query, 16), tol)
