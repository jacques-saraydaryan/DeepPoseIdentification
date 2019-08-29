import os
import glob
import argparse

# Run on Python 3 only
# The recursive mode of glob is not available on python 2

# Get external argument
parser = argparse.ArgumentParser(description="Rename all the file from a folder")
parser.add_argument('folder_path', type=str, help='Enter folder path')

# Recursive rename of files from the path parameter
def renameFiles(path):
    files = [f for f in glob.glob(path + "**/*.*",recursive=True)]

    print("Rename the following files: \n")

    for i, file_path in enumerate(files):
        newFileName = 'img' + str(i) + os.path.splitext(file_path)[1]
        dest = os.path.join(ntpath.dirname(file_path), newFileName)
        print(os.path.basename(file_path) , " -> ", newFileName)
        os.rename(file_path, dest)

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.folder_path

    renameFiles(path)
