# Project Overview

According to the [BC government](https://www2.gov.bc.ca/gov/content/safety/wildfire-status/wildfire-response/how-wildfire-is-detected), about 40% of forest fires are reported by the general public, in addition to other detection strategies such as:
- Air patrols
- Fire warden ground patrols
- Infrared technology
- Computer technology and predictive software
- Lookout towers

If not detected and managed early on, forest fires can become very destructive event that impacts both remote communities and large urban centres alike. Since it's not possible to have people in remote areas constantly monitoring, but rather autonomous vehicles (drones), satellite imagery, and optical systems, there is an opportunity to use deep learning models for image classification to enable early automated detection of fires.

Having reliable automated and early detection of fires can impact the response time and management before they become too large to control. This ensures safety for those living in immediate proximity, as well as the firefighter crews. Additionally, the financial burden on taxpayers overall could be reduced. Just last year (2023) was one of the worst fire seasons on record, resulting in an overbudget of > $700 M for the provincial government in BC ([see this news article](https://vancouver.citynews.ca/2023/09/27/bc-projected-deficit-2023-q1/)).

# Project Steps

These are the proposed Project Steps in order of complexity:

1. Setup development environment for the PyTorch library
2. Train an image classifier with same-sized fire and non-fire images. 
	1. Preprocess images to ensure they are all the same size
	2. Investigate accuracy metrics for classification tasks (accuracy, precision, recall, F1 score)
	3. Start with simplest model (i.e. LeNet)
	4. Identify state-of-the-art models that could do transfer learning (i.e. VGG, ResNet)
3. Investigate segmentation of images
4. Investigate resizing of images
	1. Evaluate what strategy to pursue for image resizing

# Datasets

In order to train image classification models, the following labelled image datasets were obtained as described below. We'll number them for easy identification

## 01 Fire Dataset

**Source:**

- https://www.kaggle.com/datasets/phylake1337/fire-dataset

**Description:** 

The dataset was created during the NASA Space Apps Challenge in 2018, with the goal being to use the dataset to develop a model that can recognize the images with fire.

**Number of images:** 999 in total

- 755 outdoor fires
- 244 non-fire images

**Image size (pixels):** 
- Variable, on average, they are 750 x 1187

**Total Dataset Size:** 
- 406 MB

**Comments:**
- There are 41 images smaller than 250 x 250, which might not be usable.
## 02 Forest Fire Dataset

**Source:** 
- https://www.kaggle.com/datasets/alik05/forest-fire-dataset
- A. Khan, B. Hassan, S. Khan, R. Ahmed and A. Adnan, “DeepFire: A Novel Dataset and Deep Transfer Learning Benchmark for Forest Fire Detection,” Mobile Information System, vol. 2022, pp. 5358359, 2022 [doi](https://doi.org/10.1155/2022/5358359).

**Description:** 

This is the dataset used for the article cited above. It was prepared by the same group as 01_Fire_Dataset, but it provides more consistent sizing with all images.

**Number of images:** 
1900 in total
- 950 fire images
- 950 non-fire images

**Image size (pixels):** 
- 250 x 250

**Total Dataset Size:** 
- 149 MB

**Comments:**
- I found 74 images that exceeded the 250 x 250 size, so I've cropped them accordingly.

## 03 The Wildfire Dataset

**Source:** 
- https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset/data
- El-Madafri I, Peña M, Olmedo-Torre N. *The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach.* Forests. 2023; 14(9):1697.  [doi](https://doi.org/10.3390/f14091697)

**Description:** 

The dataset contains 2,700 aerial and ground-based images, it has been curated from a diverse array of online platforms such as government databases, Flickr, and Unsplash. This dataset aims to capture a wide spectrum of environmental scenarios, forest variants, geographical locations, as well as confounding elements. 

There are 3 sets of images: training, test, and validation. Within each set there are 2 classes, fire/nofire, and the images are further categorized as follows.

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

**Number of images:** 
2700 images

**Image size (pixels):** 
- Variable, ranging from 

**Total Dataset Size:** 
- 11 GB

**Comments:**



# Directory Structure

`PLACEHOLDER` 
To provide details about the repository structure and its contents
```nohighlight
```


# References

 A. Khan, B. Hassan, S. Khan, R. Ahmed and A. Adnan, “DeepFire: A Novel Dataset and Deep Transfer Learning Benchmark for Forest Fire Detection,” Mobile Information System, vol. 2022, pp. 5358359, 2022 [doi](https://doi.org/10.1155/2022/5358359).
