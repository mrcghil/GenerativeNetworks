import numpy as np
import re, os, random
import cv2 #OpenCV module

__doc__ = '\n' + "Module providing some easy to use functions to combine images."

def multiple_image_import(path:str, dir_regex:str = "", last:bool = True, filter_regex:str = ""):
    # Imports images from a base path and filters them according to simple rules.
    all_subdirectories =  [x[0] for x in os.walk(path)]
    if dir_regex != "":
        all_subdirectories = [dir for dir in all_subdirectories if dir_regex in dir]
    # Now find and filter all images
    image_list = []
    for dir in all_subdirectories:
        all_images = [x[2] for x in os.walk(dir)]
        all_images = [ os.path.join(dir, img_file_name) for img_file_name in all_images[0]]
        if last == True and filter_regex == "":
            if ".yml" not in all_images[-1]:
                last_image = all_images[-1]
            else:
                last_image = all_images[-2]
            image_list.append(last_image)
        elif filter_regex != "":
            filter = re.compile(filter_regex)
            image_list += [img for img in all_images if filter.match(img)]
        else:
            print("ImageGeneration.image_combine.multiple_image_import :: Importing all!")
            image_list += all_images
    # Load all images found
    found_images = len(image_list)
    for (index, image_path) in enumerate(image_list):
        print(f"ImageGeneration.image_combine.multiple_image_import :: loading image {index + 1} of {found_images}")
        image_list[index] = cv2.imread(image_path)
    # Now return
    return image_list


def combine_images(rows: int, cols: int, image_list: list, randomize: bool = True):
    # Combines a list of images in a single image 
    image_number = rows * cols
    resized_image_list = []
    if randomize == True and len(image_list) > 1:
        print(f"ImageGeneration.image_combine.combine_images :: multiple images in a single {rows}x{cols} array (randomized order)!")
        if len(image_list) < image_number:
            # Add images to compensate
            pass
        random.shuffle(image_list)
        resized_image_list = image_list[0:image_number]
        image_stripes = []
        for i in range(cols):
            image_stripes.append(np.concatenate(resized_image_list[rows*i:rows*(i+1)], axis=0))
        image_out = np.hstack(image_stripes)
    elif randomize == False and len(image_list) > 1:
        print(f"ImageGeneration.image_combine.combine_images :: multiple images in a single {rows}x{cols} array (no order change)!")
        while len(resized_image_list) < image_number:
            resized_image_list += image_list
        resized_image_list = resized_image_list[0:image_number]
    else:
        print(f"ImageGeneration.image_combine.combine_images :: single image to repeat {rows}x{cols} times!")
        image_out = np.tile(image_list[0], (rows, cols, 1))
    return image_out


if __name__ == "__main__":
    # Single input to test combination
    single_input = "C:\\WORKSPACES\\ZINKY\\GenerativeNetworks\\Results\\Combi_Vromy_0028\\00600.png"
    single_input_image = cv2.imread(single_input)
    # Multiple import test
    multiple_input = multiple_image_import("C:\\WORKSPACES\\ZINKY\\GenerativeNetworks\\Results", dir_regex="Vromy", last=True)
    # Create a combined image from a single image
    single_combined = combine_images(rows=10, cols=10, image_list=[single_input_image], randomize=False)
    # cv2.imwrite("C:\\WORKSPACES\\ZINKY\\GenerativeNetworks\\Results\\HighRes\\single_combined.png", single_combined)
    # cv2.imshow("Single combined image", single_combined)
    # cv2.waitKey()
    # Create a combined image from multiple
    multiple_combined = combine_images(rows=12, cols=12, image_list=multiple_input, randomize=True)
    cv2.imwrite("C:\\WORKSPACES\\ZINKY\\GenerativeNetworks\\Results\\HighRes\\multiple_combined.png", multiple_combined)
    cv2.imshow("Multiple combined image", multiple_combined)
    cv2.waitKey()


