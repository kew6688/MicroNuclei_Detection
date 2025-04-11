"""
Compare the difference between using apoptosis resolve. The experiment is looking for how many apoptosis cases model met.
"""

import os
from mn_segmentation.lib.Application import Application

def compare():
    app = Application()

    pred = []

    folders = ["/content/Ulises_counts/Day 3 count", "/content/Ulises_counts/Day 6 count"]
    for i in range(2):
        folder = folders[i]
        image_paths = sorted(os.listdir(folder))

        for image_path in image_paths:
            if image_path[:2] == "._": continue
            cnt_with_apop_resolve = app.predict_image_count(os.path.join(folder,image_path),True,0.7,True)
            cnt_without_apop_resolve = app.predict_image_count(os.path.join(folder,image_path),False,0.7,True)
            if cnt_with_apop_resolve != cnt_without_apop_resolve:
                pred.append(os.path.join(folder,image_path))
    return pred

if __name__ == "__main__":
    print(compare())