import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

# Put do glavne mape projekta
ROOT_DIR = r"C:\Users\LeonardaBrkovic\Diplomski_MaskRCNN"

# Učitaj MASKRCNN mapu
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Put do težinskih koeficijenata
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Mapa za spremanje težinskih koeficijenata modela
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



class CustomConfig(Config):
    """Konfiguracija za naš skup podataka,naslijeđuje klasu roditelj,te
    mijenja parameter iz klase roditelj za naš skup podataka
    """
    NAME = "object"

    #Korišten je CPU,a li je za batch size potrebno definirati GPU
    #Trebalo bi raditi i ako se namjesti GPU, ali onda treba paziti 
    # na veličinu GPU
    IMAGES_PER_GPU = 1

    # Broj klasa (+1 za pozadinu)
    NUM_CLASSES = 1 + 2  # Pozadina + rodne grane i vodopije

    # Broj trenirajućih koraka za svaku epohu, izračun je definiran
    # kao duljina seta za trening / veličina batcha
    # imamo 265 slika za trening, a veličina batch je 1
    STEPS_PER_EPOCH = 265

    # Preskoči detekcije manje od 80%
    DETECTION_MIN_CONFIDENCE = 0.8

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Novi skup podataka, u mrcnn je definiran pas-mačka,
        mi unosimo svoj novi skup podataka
        dataset_dir: mapa u kojoj se nalazi projekt.
        subset: Subset to load: trening ili validacija?
        """
        # Dodavanje klasa.
        self.add_class("object", 1, "vodopije")
        self.add_class("object", 2, "rodne_grane")

        # Poziv će se izvršiti u ovisnosti o tome jel
        # vršimo trening ili validaciju
        assert subset in ["training_masline", "val_masline"]
        if subset=="training_masline":
            dataset_dir = os.path.join(dataset_dir, "training_masline")
            annotations1 = json.load(open(r'C:\Users\LeonardaBrkovic\Diplomski_MaskRCNN\Dataset\training_masline\region_data.json'))
        else:
            dataset_dir = os.path.join(dataset_dir, "val_masline")
            annotations1 = json.load(open(r'C:\Users\LeonardaBrkovic\Diplomski_MaskRCNN\Dataset\val_masline\region_data.json'))

        annotations = list(annotations1.values())  

        #Preskoči slike koje nisu označene
        annotations = [a for a in annotations if a['regions']]
        
        # Dodaj slike
        for a in annotations:
            # Dobavljanje x i y koordinata za svaki objekt u slici koji
            # se u JSON datoteci nalazi u :
            # shape_attributes
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['tip'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {"vodopije": 1,"rodne_grane": 2}

            num_ids = [name_dict[a] for a in objects]
     
            # load_mask() treba veličinu slike da bi moglo konvertirati poligon u maske
            # ovaj proces nije uključen u JSON datoteku stoga moramo proces ručno odraditi
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  
                image_id=a['filename'],  
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Konvertiranje poligona u bitmapu
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(r"C:\Users\LeonardaBrkovic\Diplomski_MaskRCNN\Dataset", "training_masline")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(r"C:\Users\LeonardaBrkovic\Diplomski_MaskRCNN\Dataset", "val_masline")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=15,
    #             layers='all',
    #             augmentation = iaa.Sometimes(iaa.OneOf([
    #             iaa.Fliplr(1),
    #             iaa.Flipud(1),
    #             iaa.Affine(rotate=(-45, 45)),
    #             iaa.Affine(rotate=(-90, 90)),
    #             iaa.Affine(scale=(0.5, 1.5))
    #         ])))
    # print("Training network heads 4+ layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=30,
    #             layers='4+')
    print("Treniranje samo glava mreže")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')
                          
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
        # Download weights file
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

train(model)			