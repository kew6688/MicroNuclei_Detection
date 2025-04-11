import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_crop_windows(image_path, n_wnd, row, box_sz=(224,224), stride=(224,224)):
    '''
    display the cropped image as boxes on the image.

    Args:
        image_path: str
        n_wnd: int, number of windows
        box_sz: (int,int) x,y length of the box
        stride: (int,int) x,y length of the stride
    '''
    # Load an image using PIL
    image = Image.open(image_path)

    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Define a sequence of boxes, each box is defined by (x, y, width, height)
    boxes = []

    box_w, box_h = box_sz
    x,y = stride

    for i in range(n_wnd):
        # tile image
        cur_x, cur_y = x * (i//row), y * (i%row)
        box = (cur_x, cur_y, box_w, box_h)
        boxes.append(box)

    # Draw each box on the image
    for box in boxes:
        rect = patches.Rectangle(
            (box[0], box[1]),  # (x, y)
            box[2],  # width
            box[3],  # height
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)

    # Show the final image with boxes
    plt.show()

def save_tiles(img_dir, dest_dir, n_wnd, box_sz, stride, footer=False):
    '''
    save the cropped image to dest_dir.

    Args:
        img_dir: str
        dest_dir: str
        n_wnd: int, number of windows
        box_sz: (int,int) x,y length of the box
        stride: (int,int) x,y length of the stride
        footer: bool
    '''
    box_w, box_h = box_sz
    x,y = stride
    cnt = 0

    for file in os.listdir(img_dir):
        if file[:2] == '._': continue
        im = Image.open(dir+file)
        for i in range(n_wnd):
            if footer and i in [4,9]: continue
            # tile image
            cur_x, cur_y = x * (i//5), y * (i%5)
            box = (cur_x, cur_y, cur_x + box_w, cur_y + box_h)
            img2 = im.crop(box)

            img2.save(dest_dir + file.split('.')[0] + '-' + str(i) + '.png')
            cnt += 1
    print(cnt)