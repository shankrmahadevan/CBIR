# CBIR
A Personal Reverse Search Engine Inspired By Google's own Content based image retrieval image system, that can be used in our own desktop. It highly leverages the Ray Multiprocessing library to fastly encode the images into feature vectors, to make them searchable. Highly Vectorized Search algorithm retrieves result within one second(when tested on a collection of 100k images)

When run for the first time, it takes time to encode the images present in the folder that is being searched. But, it is blazing-fast once all the images have been encoded. Make sure not to delete the pickle file (.pkl) that appears on the search-folder once u run this code.

Setup:
Just run SearchEngine.py.
