from detectron2.data import DatasetCatalog

"""
Your function should do arbitrary thing, but return a list[dict]
Where each dict contains information of an image
You must ensure the function returns the exactly same thing each time it is called
For the format of list[dict], see Detectron2's official document:
https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

Generally, each dict should contain following keys:
file_name (str): the full path to your image
height (int): the height of input image
width (int): the width of input image
image_id (str or int): a unique id appointed to a specific image
sem_seg_file_name (str): the full path to your semantic segmentation masks, where for each mask,
                        you need to use a pixel value for a class, starting from 0, 1, ...
"""

def load_train_data():
    # Write your own function to load data
    pass


def load_val_data():
    # Write your own function to load data
    pass


DatasetCatalog.register("MyDataset_Train", load_train_data)
DatasetCatalog.register("MyDataset_Test", load_val_data)

