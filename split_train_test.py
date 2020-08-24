import argparse
import pathlib
import os
import zipfile
import random

parser = argparse.ArgumentParser(description='Input parameters for data split')
parser.add_argument('--files_zip', type=str,
                    help='the zip file that contains the folder with images', required=True)
parser.add_argument('--new_dir_name', type=str,
                    help='renamed the directory here the files will be unzipped', required=True)
parser.add_argument('--split_tags', type=str, nargs='*',
                    help='tags for the files to be split on', required=True)

args = parser.parse_args()

files_zip = str(args.files_zip)
extra_path = files_zip.strip('.zip')
new_dir_name = str(args.new_dir_name)
split_tags = args.split_tags


if os.path.exists(args.files_zip):
    if os.path.exists(extra_path):
        print('Remove all folders with the name as the archive')
    else:
        if os.path.exists(new_dir_name):
            print('There is already a folder with the same name as new_dir_name')
        else:
            with zipfile.ZipFile(args.files_zip, mode='r') as zp:
                   zp.extractall()
            # os.rename(extra_path, new_dir_name)

            # set the directory where the images will be found
            data_dir = pathlib.Path(extra_path)

            # print(data_dir)

            # remove not jpg files
            other_files = list(data_dir.glob('*/*.db'))

            # print(other_files)

            for x in other_files:
                os.remove(str(x))

            # take the folder names
            fold_names = list(data_dir.glob('*/'))
            # print(fold_names)

            # separate the folders by open or close. It is specified in the name of the folder

            if len(split_tags) <= 1:
                print('Must supply at least 2 tags')
            else:
                dict_tags = {}
                for tag in split_tags:
                    dict_tags[tag] = [fold for fold in fold_names if str.find(str(fold), tag) != -1]

                non_tags = []
                for k, v in dict_tags.items():
                    if len(v) == 0:
                        non_tags.append(v)

                if len(non_tags) != 0:
                    print(f'The following tags were not found in folder names: {non_tags}')
                else:
                    # take a random choice of files that will go in the valid folder
                    rand_chose = {}

                    for fold in fold_names:
                        pict = list(fold.glob('*.jpg'))
                        rand_chose[fold] = random.choices(pict, k=10)

                    # create train and valid folder and inside them all the 2 folder above (open close) to have the same structure as parent
                    train_dir = pathlib.Path(data_dir, 'train')
                    valid_dir = pathlib.Path(data_dir, 'valid')

                    os.mkdir(str(pathlib.Path(data_dir, 'train')))
                    os.mkdir(str(pathlib.Path(data_dir, 'valid')))

                    for f in split_tags:
                        os.mkdir(str(pathlib.Path(data_dir, 'train', f)))
                        os.mkdir(str(pathlib.Path(data_dir, 'valid', f)))

                    # move random choice files in valid
                    for fold in rand_chose.keys():
                        for pic in rand_chose[fold]:
                            if str.find(str(pic), split_tags[0]) != -1:
                                os.rename(pic, str(pathlib.Path(valid_dir, split_tags[0], str.split(str(pic), '\\')[-1])))
                            else:
                                os.rename(pic, str(pathlib.Path(valid_dir, split_tags[1], str.split(str(pic), '\\')[-1])))


                    # move the rest of files in train
                    for fold in fold_names:
                        pics = list(fold.glob('*.jpg'))
                        for pic in pics:
                            if str.find(str(pic), split_tags[0]) != -1:
                                os.rename(pic, str(pathlib.Path(train_dir, split_tags[0], str.split(str(pic), '\\')[-1])))
                            else:
                                os.rename(pic, str(pathlib.Path(train_dir, split_tags[1], str.split(str(pic), '\\')[-1])))

                    # delete the empty folders
                    for fold in fold_names:
                        os.removedirs(fold)

                    # change the name of the final folder
                    os.rename(extra_path, '_'.join(split_tags))

                    count = len(list(data_dir.glob('*/*/*.jpg')))
                    print(f'The number of all images is {count}')
