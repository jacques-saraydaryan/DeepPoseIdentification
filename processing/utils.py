import os
import glob
import argparse

# Run on Python 3 only
# The recursive mode of glob is not available on python 2


# Recursive rename of files from the path parameter
def renameFiles(path, mask):
    files = [f for f in glob.glob(path + "**/*.*",recursive=True)]

    print("Rename the following files: \n")
    if mask:
        mask = mask + '_'

    for i, file_path in enumerate(files):
        newFileName = 'img_' + mask + str(i) + os.path.splitext(file_path)[1]
        dest = os.path.join(os.path.dirname(file_path), newFileName)
        print(os.path.basename(file_path) , " -> ", newFileName)
        os.rename(file_path, dest)

if __name__ == '__main__':
    # Get external argument
    parser = argparse.ArgumentParser(description="Rename all the file from a folder")
    parser.add_argument('folder_path', type=str, help='Enter folder path')
    parser.add_argument('--mask', type=str, default="", help='Enter a mask to add in the name of your images')
    args = parser.parse_args()

    renameFiles(args.folder_path, args.mask)
