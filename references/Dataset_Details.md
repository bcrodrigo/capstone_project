In order to train image classification models, the following labelled image datasets were obtained as described below. We'll number them for easy identification
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


## 02 Forest Fire Dataset

**Source:** 
- https://www.kaggle.com/datasets/alik05/forest-fire-dataset
- A. Khan, B. Hassan, S. Khan, R. Ahmed and A. Adnan, “DeepFire: A Novel Dataset and Deep Transfer Learning Benchmark for Forest Fire Detection,” Mobile Information System, vol. 2022, pp. 5358359, 2022 [doi](https://doi.org/10.1155/2022/5358359).

**Description:** 

This is the dataset used for the article cited above. It was prepared by the same group as `01_fire_dataset`, but it provides more consistent sizing with all images.

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
- El-Madafri I, Peña M, Olmedo-Torre N. *The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach.* Forests. 2023; 14(9):1697.  [doi](https://doi.org/10.3390/f14091697)

**Description:** 

The dataset contains 2,700 aerial and ground-based images. The authors curated it from a diverse array of online platforms such as government databases, Flickr, and Unsplash. This dataset aims to capture a wide spectrum of environmental scenarios, forest variants, geographical locations, as well as confounding elements. 

There are 3 sets of images: training, test, and validation. Within each set there are 2 classes, fire/nofire, and the images are further categorized as follows

```bash
.
├── fire
│   ├── Both_smoke_and_fire
│   └── Smoke_from_fires
└── nofire
    ├── Fire_confounding_elements
    ├── Forested_areas_without_confounding_elements
    └── Smoke_confounding_elements
```


**Number of images:** 2700 images in total with
- 40% fire
- 60% non-fire

**Image size (pixels):** Variable

**Total Dataset Size:** 11 GB

**Comments:**
- Will likely need to resize this dataset to a consistent size that can be easily transformed (250 x 250)