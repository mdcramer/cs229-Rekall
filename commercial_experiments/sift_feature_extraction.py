import cv2
import glob
import imageio
import numpy as np
import os

# hyperparameters
input_data_directory = 'C:\_data\Commercials\images'
output_file = 'C:\_data\Commercials\data\num_features.txt'
num_features = 1000000  # keep this number large, orb finds a lot less features than this number


def read_data_from_input_directory():
    paths = []
    # collect child directories
    directory_paths = next(os.walk(input_data_directory))[1]
    # collects paths to frames in every child directory
    for directory in directory_paths:
        paths += glob.glob(input_data_directory + directory + "/*.jpg")
    return paths, len(paths)


def get_num_keypoints(orb, image_matrix):
    # computes and returns number of keypoints found in an frame/image_matrix
    # orb.detect returns keypoints, we only need its length
    return len(orb.detect(image_matrix, None))


if __name__ == "__main__":
    # get paths to individual frames
    data_paths, num_files = read_data_from_input_directory()
    # create orb object
    orb = cv2.ORB_create(nfeatures=num_features)
    # make numpy array
    num_keypoints = np.zeros(num_files)
    # get number of features
    for index, file_path in enumerate(data_paths):
        image = imageio.imread(file_path)
        num_keypoints[index] = get_num_keypoints(orb, image)
    # save number of features to a txt file
    np.savetxt(output_file, num_keypoints)
