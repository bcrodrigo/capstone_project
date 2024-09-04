# Dataset Details
The following labelled image datasets were obtained as described in the sections below. We'll number all datasets for easy identification.

Please note that:
- Dataset `01` was only used for EDA purposes.
- Datasets `02` and `03` were used for training and testing.

## 01 Fire Dataset

**Source:**
- https://www.kaggle.com/datasets/phylake1337/fire-dataset

**Description:** 
The dataset was created during the NASA Space Apps Challenge in 2018, with the goal being to use the dataset to develop a model that can recognize the images with fire.

**Number of images:** 999 in total, this includes
- 755 outdoor fires (75%)
- 244 non-fire images (25%)

**Image size (pixels):** Variable, on average, they are 750 x 1187

**Total Dataset Size:** 406 MB

**Comments:**
- There are 41 images smaller than 250 x 250, which might not be usable.

## 02 Forest Fire Dataset (DeepFire)

**Source:** 
- https://www.kaggle.com/datasets/alik05/forest-fire-dataset

**Description:** 
This is the dataset used in reference [^1], where VGG19 (with transfer learning) is compared against other machine learning models (logistic regression, naive Bayes, random forests, SVM, and KNN). The dataset was prepared by the same group as `01_fire_dataset`, but it provides more consistent sizing with all images.

**Number of images:** 1900 in total
- 950 fire images
- 950 non-fire images

**Image size (pixels):** 250 x 250

**Total Dataset Size:** 149 MB

**Comments:**
- I found 74 images that exceeded the 250 x 250 size, so I've cropped them accordingly.

## 03 The Wildfire Dataset

**Source:** 
- https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset/data

**Description:** 
The dataset contains 2,700 aerial and ground-based images. The authors curated it from a diverse array of online platforms such as government databases, Flickr, and Unsplash. This dataset aims to capture a wide spectrum of environmental scenarios, forest variants, geographical locations, as well as confounding elements. 

There are 3 sets of images: training, test, and validation. Within each set there are 2 classes: "fire" and "nofire" so that the images are further categorized as follows
```bash
.
├── fire/
│   ├── Both_smoke_and_fire/
│   └── Smoke_from_fires/
└── nofire/
    ├── Fire_confounding_elements/
    ├── Forested_areas_without_confounding_elements/
    └── Smoke_confounding_elements/
```

The dataset is used in reference [^2], to train MobileNetV3 under different strategies. Note that in the publication, the images are organized in the folder structure detailed above. The kaggle link however, only provides 3 folders with train test and validation images.

**Number of images:** 2700 images in total with
- 40% fire
- 60% non-fire

**Image size (pixels):** Variable

**Total Dataset Size:** 11 GB

**Comments:**
- In order to train models, I resized all images to 250 x 250 pixels.

## References
[^1]: A. Khan, B. Hassan, S. Khan, R. Ahmed and A. Adnan, *DeepFire: A Novel Dataset and Deep Transfer Learning Benchmark for Forest Fire Detection* Mobile Information System, vol. 2022, pp. 5358359, 2022 [doi](https://doi.org/10.1155/2022/5358359).
[^2]: El-Madafri I, Peña M, Olmedo-Torre N. *The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach.* Forests. 2023; 14(9):1697.  [doi](https://doi.org/10.3390/f14091697)