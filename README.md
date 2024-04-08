# Project Overview

According to the [BC government](https://www2.gov.bc.ca/gov/content/safety/wildfire-status/wildfire-response/how-wildfire-is-detected), about 40% of forest fires are reported by the general public, in addition to other detection strategies such as:
- Air patrols
- Fire warden ground patrols
- Infrared technology
- Computer technology and predictive software
- Lookout towers

If not detected and managed early on, forest fires can become a very destructive event that impacts both remote communities and large urban centres alike. Considering the vastness of the territory in the province of BC and the low population density, it's not possible to have people in remote areas constantly monitoring, but rather autonomous vehicles (drones), satellite imagery, and optical systems. With the latter, there is an opportunity to use Deep Learning (DL) models for image classification and segmentation to enable early automated detection of fires. Some authors have reported 95% accuracy scores for DL models classifying forest fire images (see [doi](https://doi.org/10.1155/2022/5358359)).

Having reliable automated and early detection of fires can impact the response time and management before they become too large to control. This ensures safety for those living in immediate proximity, as well as the firefighter crews. Additionally, the financial burden on taxpayers overall could be reduced. Last year (2023) was the worst fire seasons on record, resulting in an overbudget of > $700 M for the provincial government in BC ([see this news article](https://vancouver.citynews.ca/2023/09/27/bc-projected-deficit-2023-q1/)).

# Project Steps

These are the proposed Project Steps in order of complexity:

1. Setup development environment for the PyTorch library
	- [x] Setup `DataLoader` class
	- [x] Organize Python modules in `src/` directory
2. Train an image classifier with same-sized fire and non-fire images
	- [x] Preprocess `02_fire_dataset` images to ensure they are all the same size
	- [x] Investigate accuracy metrics for classification tasks (accuracy, precision, recall, F1 score)
	- [x] Implement simplest CNN model (i.e. LeNet)
	- [x] Identify state-of-the-art models that could do transfer learning (i.e. VGG, ResNet)
	- [x] Using `02_fire_dataset` train VGG19 with transfer learning, using the same hyperparameters as [authors](https://doi.org/10.1155/2022/5358359). 
	- [x] Train ResNet18 with `02_fire_dataset` using same hyperparameters as [authors](https://doi.org/10.1155/2022/5358359). 
3. Investigate segmentation of images
	- [x] Item cancelled, as it is out of scope. Will have to revisit for a future project.
4. Make a new version of the `03_the_wildfire_dataset`
	- [x] Convert to square images of 250 x 250
5. Compare VGG19 and ResNet
	- [ ] What is the nature of false positives and false negatives?
	- [ ] What is the most important metric for real-world deployment of these models?
	- [ ] How do models perform doing predictions from images with confounding elements (smoke, sunlight) from `03_the_wildfire_dataset` ?
	- [ ] Can the models be re-trained with the `03_the_wildfire_dataset` and still perform well on unseen data?


# Datasets

Follow the following ling for [[Dataset_Details]].



# Directory Structure

TO DO: Provide some details about folder contents

```nohighlight
.
├── README.md
├── LICENSE
├── jupyter_notebooks/
├── reports/
└── src/
    ├── data/
    ├── models/
    └── visualization/
```

# References

1. A. Khan, B. Hassan, S. Khan, R. Ahmed and A. Adnan, *DeepFire: A Novel Dataset and Deep Transfer Learning Benchmark for Forest Fire Detection* Mobile Information System, vol. 2022, pp. 5358359, 2022 [doi](https://doi.org/10.1155/2022/5358359).
2. El-Madafri I, Peña M, Olmedo-Torre N. *The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach.* Forests. 2023; 14(9):1697.  [doi](https://doi.org/10.3390/f14091697)