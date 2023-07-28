""" This file is for loading the relevant dataset_id

    init needs the following:
        root_path: this is the root path to the shell_dataset folder
        dataset_id: integer dataset/task id
        set_type: string 
            'train': training
            'validation': validation
            'test': test (FINAL, don't use until after all hyperparameter tuning)
        input_type: string
            'original': original image
            'cropped': cropped image (original image that was scaled and cropped to be 299x299)
            'features': feature vector (feature vector of cropped image (with normalization) passing through backbone)
        pipeline: OPTIONAL a torchvision transform pipeline

    If this file is run directly, it will test for the existence of all images for each dataset,
    for each type of input.

    helpful instance methods:
        unique_class_id_list
            list of every class_id for the current dataset/set/limits combination
        class_id_counts
            dict of every class_id and the counts of the number of images/vectors

    helpful static methods:
        verify_all_original_images_are_in_database
            Checks that all images in original_images folder are cataloged in the database.
        verify_everything
            Goes through all datasets, and verifies that that every original image, cropped image,
            and feature vector exists.
        get_list_of_datasets
            Returns a list of all datasets in database
"""


import os
import sqlite3
from dataclasses import dataclass
from glob import glob
from operator import index
from typing import Any, Optional

import h5py
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional
from torchvision import transforms

import random

SET_TYPES = ["train", "validation", "test"]
INPUT_TYPES = ["original", "cropped", "features"]
transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

# list of valid image extensions used when verifying all original files are in the database
VALID_IMAGE_EXTENSIONS = [
    "bmp",
    "gif",
    "jfif",
    "jpeg",
    "jpg",
    "pgm",
    "png",
    "ppm",
    "svg",
    "tif",
    "webp",
]

@dataclass
class ShellImage:
    hash: str = ""
    class_id: int = -1
    path: str = ""
    filename: str = ""
    feature_vector: Tensor = Tensor([])


class ShellDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        dataset_id: int,
        set_type: str,
        input_type: str,
        threshold: int = 60,
        large_size: int = 512,
        small_size: int = 256,
        random_seed: int = 27,
        label_dict = None,
        pipeline = transform,
        verify_all_sets_all_types: bool = False,
    ) -> None:
        super().__init__()
        """init function for the dataset"""

        random.seed(random_seed)

        self.initialized_correctly: bool = False

        # check inputs
        root_path = ShellDataset.sanitize_and_check_root_path(root_path)
        assert dataset_id >= 0
        assert set_type in SET_TYPES
        assert input_type in INPUT_TYPES

        # store inputs
        self.root_path = root_path
        self.dataset_id = dataset_id
        self.set_type = set_type
        self.input_type = input_type
        self.pipeline = pipeline

        # get the dataset_name
        query = f"select dataset_name FROM datasets where dataset_id={dataset_id};"
        result = ShellDataset.execute_database_query(root_path, query)
        if len(result) == 0:
            print("no dataset with that id")
            return None

        self.dataset_name = result[0][0]

        # grab the correct rows
        # print("load relevant results from database...")
        set_type_int = SET_TYPES.index(self.set_type)
        if not verify_all_sets_all_types:
            query = f'SELECT hash, path, filename, class_id FROM files WHERE dataset_id={dataset_id} AND problem=0 AND "set"={set_type_int};'
        else:
            query = f"SELECT hash, path, filename, class_id FROM files WHERE dataset_id={dataset_id} AND problem==0;"
        rows = ShellDataset.execute_database_query(root_path, query)

        if len(rows) == 0:
            print(f"no results for dataset_id={dataset_id} and set_type={set_type_int}")
            return None
        # store the details for the images
        self.images: list[ShellImage] = []
        self.dict = {}
        self.label_dict = {}
        for row in rows:
            if int(row[3]) not in self.dict:
                self.dict[int(row[3])] = [(row[0], row[1], row[2])]
            else:
                self.dict[int(row[3])].append((row[0], row[1], row[2]))
        label_use = 0
        if label_dict:
            self.label_dict = label_dict
        else:
            for label in self.dict.keys():
                self.label_dict[label] = label_use
                label_use += 1

        if len(self.dict.keys()) > 300:
            self.num_classes = 300
        else:
            self.num_classes = len(self.dict.keys())

        if len(self.dict.keys()) >= threshold and set_type_int == 0:
            self.labels, self.hashes, self.paths, self.filenames = self._random_value(self.dict, large_size*10)
        elif len(self.dict.keys()) < threshold and set_type_int == 0:
            self.labels, self.hashes, self.paths, self.filenames = self._random_value(self.dict, small_size*10)
        elif len(self.dict.keys()) >= threshold and set_type_int > 0:
            self.labels, self.hashes, self.paths, self.filenames = self._random_value(self.dict, large_size)
        elif len(self.dict.keys()) < threshold and set_type_int > 0:
            self.labels, self.hashes, self.paths, self.filenames = self._random_value(self.dict, small_size)
        for i in range(len(self.labels)):
            self.labels[i] = self.label_dict[self.labels[i]]
            self.images.append(ShellImage(hash=self.hashes[i], class_id=self.labels[i], path=self.paths[i], filename=self.filenames[i]))

        if not verify_all_sets_all_types:
            if self.input_type == "original":
                self.initialized_correctly = self.verify_original_images_exist()
            elif self.input_type == "cropped":
                self.initialized_correctly = self.verify_cropped_images_exist()
            else:  # features input
                self.initialized_correctly = (
                    self.verify_feature_vectors_exist_and_load()
                )
        else:
            # assume true, but set to false if any fail
            self.initialized_correctly = True
            if not self.verify_original_images_exist():
                self.initialized_correctly = False
            if not self.verify_cropped_images_exist():
                self.initialized_correctly = False
            if not self.verify_feature_vectors_exist_and_load(skip_load=True):
                self.initialized_correctly = False

    def _random_value(self, dictionary, amount):
        label = []
        hash = []
        path = []
        filename = []
        num_each_class = round(amount/self.num_classes)

        current_class_size = 0
        for key, values in dictionary.items():
            track_num = 0
            if key not in list(self.label_dict.keys())[:self.num_classes]:
                assert self.num_classes == 300
                if self.num_classes == 300:
                    continue
            random.shuffle(values)
            for value in values:
                label.append(key)
                hash.append(value[0])
                path.append(value[1])
                filename.append(value[2])
                track_num += 1
                if track_num >= num_each_class:
                    break
            current_class_size += 1
            if current_class_size >= self.num_classes:
                break
        return label, hash, path, filename

    def __len__(self) -> int:
        assert self.initialized_correctly

        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        assert self.initialized_correctly
        assert 0 <= index <= len(self.images)

        if self.input_type == "features":
            return self.images[index].feature_vector, self.images[index].class_id
        else:
            if self.input_type == "original":
                image_filename = self.original_image_path_for_index(index)
            else:
                image_filename = self.cropped_image_path_for_index(index)

            with Image.open(image_filename) as image:

                # for the original images, make sure to convert to RGB
                # cropped images already have this done
                if self.input_type == "original":
                    image = image.convert("RGB")

                # convert to a tensor before doing anything else
                # image_tensor = functional.to_tensor(image)

            # if there is a pipeline, run it
            if self.pipeline:
                image_tensor = self.pipeline(image)

            return image_tensor, self.images[index].class_id, self.dataset_id

    def original_image_path_for_index(self, index: int) -> str:
        """get the path to the original image"""
        assert 0 <= index <= len(self.images)

        return (
            self.root_path
            + "original_images/"
            + self.dataset_name
            + self.images[index].path
            + self.images[index].filename
        )

    def cropped_image_path_for_index(self, index: int) -> str:
        """get the path for the cropped image"""
        assert 0 <= index <= len(self.images)
        hash = self.images[index].hash

        return (
            self.root_path
            + "cropped_images/"
            + hash[0]
            + "/"
            + hash[1]
            + "/"
            + hash
            + ".webp"
        )

    def verify_original_images_exist(self) -> bool:
        """check that all original image files exist"""

        had_error = False
        # print("verifying that original images exist...")

        # order is not important
        for index in range(len(self.images)):
            original_filename = self.original_image_path_for_index(index)
            if not os.path.isfile(original_filename):
                print("missing", original_filename)
                had_error = True

        return not had_error

    def verify_cropped_images_exist(self) -> bool:
        """check that all cropped image files exist"""

        had_error = False
        # print("verifying that cropped images exist...")

        # order is not important
        for index in range(len(self.images)):
            cropped_filename = self.cropped_image_path_for_index(index)
            if not os.path.isfile(cropped_filename):
                print("missing", cropped_filename)
                had_error = True

        return not had_error

    def verify_feature_vectors_exist_and_load(self, skip_load: bool = False) -> bool:
        """check that the hdf5 file exists, and load all of the feature vectors for the set"""

        had_error = True
        feature_vectors_filename = (
            self.root_path + "feature_vectors/" + self.dataset_name + ".h5"
        )

        # if not skip_load:
        #     print("verifying all feature vectors exist, and loading...")
        # else:
        #     print("verifying all feature vectors exist...")

        if not os.path.isfile(feature_vectors_filename):
            print("feature vector file is missing", feature_vectors_filename)
        else:

            all_hashes = []
            # force order to be the same
            for i in range(len(self.images)):
                all_hashes += [self.images[i].hash]

            with h5py.File(feature_vectors_filename) as file_h:
                # check that all hashes exist in file
                feature_vectors_hashes = list(file_h.keys())
                missing_list = list(set(all_hashes) - set(feature_vectors_hashes))

                if len(missing_list) > 0:
                    print("missing these feature vectors:", missing_list)
                else:
                    # if no issues, create matrix of feature vectors
                    if not skip_load:
                        for i, _ in enumerate(self.images):
                            self.images[i].feature_vector = file_h[self.images[i].hash][:]  # type: ignore
                    had_error = False

        return not had_error

    def get_entry_for_index(self, index: int) -> ShellImage:
        """get the entry for a given index"""
        assert self.initialized_correctly
        assert 0 <= index <= len(self.images)

        return self.images[index]

    @property
    def unique_class_id_list(self) -> list:
        """get a list of unique class_id"""
        assert len(self.images) > 0

        all_class_ids = [x.class_id for x in self.images]
        # return an ordered and unique list
        return list(set(all_class_ids))

    @property
    def class_id_counts(self) -> dict:
        """get the counts for all class_ids"""
        assert len(self.images) > 0

        counts = {}
        all_class_ids = [x.class_id for x in self.images]
        for class_id in self.unique_class_id_list:
            counts.update({class_id: all_class_ids.count(class_id)})

        return counts

    @staticmethod
    def sanitize_and_check_root_path(root_path: str) -> str:
        """make sure root path ends in '/', and that it exists"""
        root_path = (root_path + "/").replace("//", "/")
        assert os.path.isdir(root_path)

        return root_path

    @staticmethod
    def execute_database_query(root_path: str, query: str) -> list:
        """execute a query on the database at the path"""

        root_path = ShellDataset.sanitize_and_check_root_path(root_path)
        database_filename = root_path + "rolling_releases/database_2.sqlite"

        assert os.path.isfile(database_filename)

        with sqlite3.connect(database_filename) as sql_conn:
            cursor = sql_conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

        return rows

    @staticmethod
    def verify_everything(root_path: str) -> bool:
        """verify the existence of all problem-free files"""
        # print("verify all sets all input types")

        root_path = ShellDataset.sanitize_and_check_root_path(root_path)

        # get all of the dataset_ids that are in files
        # this avoids the issue of the bad dataset_ids
        query = "select distinct dataset_id FROM files order by dataset_id;"
        rows = ShellDataset.execute_database_query(root_path, query)
        dataset_id_list = [int(id[0]) for id in rows]

        broken_dataset_list = []

        # loop through each dataset
        for dataset_id in dataset_id_list:
            # print(f"#### dataset_id: {dataset_id} ####")
            shell_dataset = ShellDataset(
                root_path,
                dataset_id,
                "train",
                "original",
                verify_all_sets_all_types=True,
            )

            if not shell_dataset.initialized_correctly:
                broken_dataset_list += [dataset_id]

        if len(broken_dataset_list) == 0:
            # print("")
            # print("########################")
            # print("####    ALL GOOD    ####")
            # print("########################")
            return True
        else:
            print("")
            print("######################################")
            print("####    SOME FILES ARE MISSING    ####")
            print("######################################")
            print("datasets with issues:", broken_dataset_list)
            return False

    @staticmethod
    def get_list_of_datasets(
        root_path: str, only_good: bool = False, get_counts: bool = False
    ) -> dict:
        """get list of all datasets in database
        it returns a dict with the key as the dataset_id, and the value
        is another dict with the name of the dataset and the good flag"""
        root_path = ShellDataset.sanitize_and_check_root_path(root_path)

        if only_good:
            query = "select distinct dataset_id, dataset_name, good FROM datasets WHERE good=1 order by dataset_id;"
        else:
            query = "select distinct dataset_id, dataset_name, good FROM datasets order by dataset_id;"

        rows = ShellDataset.execute_database_query(root_path, query)

        shell_datasets_dict = {}
        for row in rows:
            dataset_id = int(row[0])

            if not get_counts:
                shell_datasets_dict.update(
                    {
                        dataset_id: {
                            "name": row[1],
                            "good": row[2],
                        }
                    }
                )
            else:
                # query all of the counts at the same time
                query = f'select count(*) FROM files WHERE problem=0 and dataset_id={dataset_id} group by "set";'
                count_rows = ShellDataset.execute_database_query(root_path, query)
                if len(count_rows) == 3:
                    set_counts = []
                    for set_i in range(3):
                        set_counts += [count_rows[set_i][0]]
                else:
                    set_counts = [0, 0, 0]
                total_count = sum(set_counts)

                shell_datasets_dict.update(
                    {
                        dataset_id: {
                            "name": row[1],
                            "good": row[2],
                            "total_count": total_count,
                            "train_count": set_counts[0],
                            "validation_count": set_counts[1],
                            "test_count": set_counts[2],
                        }
                    }
                )

        return shell_datasets_dict

    @staticmethod
    def verify_all_original_images_are_in_database(root_path: str) -> bool:
        """check that every image in original_images is cataloged in the database"""

        root_path = ShellDataset.sanitize_and_check_root_path(root_path)

        # print("begin cataloging all files in original_images...")
        # print("this will take several minutes")
        source_dir = root_path + "original_images/"
        all_files = glob("**/*.*", root_dir=source_dir, recursive=True)

        # loop through all files, and keep only those that are images (extension match)
        image_files = []
        for file in all_files:

            # get just the filename (this will also catch folders)
            after_final_slash = file.split("/")[-1]

            # first, check if possibly an image file (has an extension)
            # ignore all folders and files without an '.' in the name
            if after_final_slash.find(".") > -1:

                # then, get the extension
                extension = after_final_slash.split(".")[-1].lower()

                # check if extension is one of the valid image extension
                # ignore all other file types
                if extension in VALID_IMAGE_EXTENSIONS:
                    image_files += [file]

        # print("done")
        # print("images found in original_images folder:", len(image_files))

        # get all of the files in the database
        query = "select dataset_name || path || filename FROM files, datasets where files.dataset_id=datasets.dataset_id;"
        rows = ShellDataset.execute_database_query(root_path, query)
        db_images = [row[0] for row in rows]
        # print("images in database:", len(db_images))

        images_not_in_db = list(set(image_files) - set(db_images))
        db_images_without_files = list(set(db_images) - set(image_files))
        if (len(images_not_in_db) == 0) and (len(db_images_without_files) == 0):
            # print(
            #     "all original images exist in database, and database contains no extras"
            # )
            return True

        # print("count of original images not in database", len(images_not_in_db))
        # print("count of extra images in database", len(db_images_without_files))

        return False


if __name__ == "__main__":
    pass