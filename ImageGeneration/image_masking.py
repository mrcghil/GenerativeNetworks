import numpy as np
import cv2 #OpenCV module

__doc__ = '\n' + "Module providing some easy to use functions to trim images."

borderType = cv2.BORDER_CONSTANT
defaultFont = cv2.FONT_HERSHEY_COMPLEX

def add_border(image_in, border_percentage: float = 0.05, border_color: list = [0, 0, 0]):
    top = bottom = int(border_percentage * image_in.shape[0])
    left = right = int(border_percentage * image_in.shape[1])
    image_out = cv2.copyMakeBorder(image_in, top, bottom, left, right, borderType, None, border_color)
    return image_out

def add_alpha_channel(image_in, transparency: float = 1):
    transparency_int = transparency * 255
    alpha = np.full(image_in.shape[0:2], transparency_int, dtype=np.uint8)
    image_out = np.dstack((image_in, alpha))
    return image_out

def add_text(image_in, text: str = '', fontFace = defaultFont, fontScale: int = 0.5, thickness: int = 2, color: tuple = (255, 255, 255), location: str = 'BottomLeft', origin: tuple = ()):
    # In CV2: Origin is in the TOP-LEFT corner.
    # X_Positive to the RIGHT
    # Y_Positive DOWNWARDS
    #
    # lineType allows to specify if there is smoothing/interpolation (LINE_AA) or not (LINE_4)

    # Skip if there is no text
    if text != '':
        # Origin can be specified manually
        if origin == ():
            std_border = 0.05
            # default 'BottomLeft'
            min_distance = min(int(std_border * image_in.shape[1]), int(std_border * image_in.shape[0]))
            org = (min_distance, image_in.shape[0] - min_distance)
            # Get text size
            textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)
            if location == 'BottomRight':
                org = (int(image_in.shape[1] - textsize[0][0] - min_distance), int(image_in.shape[0] - min_distance))
            else:
                print('ImageGeneration.image_masking :: origin of text to BottomLeft')
        else:
            org = origin
        print(org)
        image_out = cv2.putText(image_in, text, org, fontFace, fontScale, color, thickness, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
        return image_out
    else:
        # No text added
        return image_in


def circular_mask():
    pass

def square_mask():
    pass

def xor_mask():
    pass




if __name__ == "__main__":
    input_image = "C:\WORKSPACES\ZINKY\GenerativeNetworks\InputFiles\ImageCollection\giraffe.jpg"
    image = cv2.imread(input_image)
    print(f'ImageGeneration.image_masking :: image shape = [{image.shape[0]}, {image.shape[1]}].')
    if image is None:
        raise('ImageGeneration.image_masking :: Cannot open the input image.')
    # Test Border Addition
    image_bordered = add_border(image)
    # cv2.imshow("Input Image with border", image_bordered)
    # cv2.waitKey()
    # Test Add Transparency
    image_semi_tranparent = add_alpha_channel(image, transparency = 0.3)
    # cv2.imwrite("./transparent_img.png", image_semi_tranparent) # Write to verify because imshow removes extra dimensions
    # Interpolated border 
    # Test Add Text
    image_signed = add_text(image, text = 'AI Artist', color = (0, 0, 0), thickness = 1, origin = (), location='BottomRight')
    # cv2.imshow("Input image with text", image_signed)
    # cv2.waitKey()

    # Useful breakpoint
    print('ImageGeneration.image_masking :: End of test!')
    # closing all open windows
    cv2.destroyAllWindows()

