import json
import os 
from PIL import Image
from collections import defaultdict
from mn_classification.src.main import *
import numpy as np

def crop_ROI(self, model_path=None):
    '''Generate smaller image for better annotation. pre-predict the image if contain mn by using pre-trained ResNet on NucRec dataset,
        to mimic the process of ROI (Region of Interest, https://www.nature.com/articles/s41586-023-06157-7#Sec60)

    Args:
        model: pre trained model that used to classify the proposed image
    
    Returns:
        None
        (print count of generated image, saved to self.crop_dir)
    '''
    model = MNClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model.to(device)

    preprocess = v2.Compose([
        # v2.Resize(size = (224,224)),
        # v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # crop image into 6 rows, 5 cols
    box_w, box_h = 224, 224
    x,y = 200, 204
    cnt = 0

    for file in os.listdir(self.dir):
        im = Image.open(self.dir+file)
        for i in range(30):
            # tile image
            cur_x, cur_y = x * (i//5), y * (i%5)
            box = (cur_x, cur_y, cur_x + box_w, cur_y + box_h)
            img2 = im.crop(box)

            # preprocess the input
            input_arr = np.array(img2)
            # print(input_arr.shape)
            input_tensor = preprocess(input_arr)

            # display preprocessed test image
            # plt.imshow(torch.permute(input_tensor, (1,2,0)))
            # plt.savefig(f"test_transformed.png")

            input_batch = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_batch)
                if output.argmax(1) == 0:
                    img2.save(self.crop_dir + file.split('.')[0] + '-' + str(i) + '.png')
                    cnt += 1
    print(cnt)

if __name__ == "__main__":
    dir = '/home/y3229wan/scratch/unlabelled/images/'
    crop_dir = '/home/y3229wan/scratch/best_ROIfilter/'
    # crop_dir = '/home/y3229wan/projects/def-sushant/y3229wan/mn-project/Data/KateData/cropped_images/'
    label = '/home/y3229wan/scratch/KateData/result.json'
    model_path = '/home/y3229wan/projects/def-sushant/y3229wan/mn-project/MN/output/best.pt'

    cropimg = CropImg(dir, crop_dir, label)
    cropimg.crop_ROI(model_path)
