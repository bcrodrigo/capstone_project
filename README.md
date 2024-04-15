# Image Classification of Forest Fires with Deep Neural Networks
In this project I use pre-trained Computer Vision Models to perform image classification of Forest Fires. For more details about the project motivation see [Project Motivation](https://github.com/bcrodrigo/capstone_project/blob/main/reports/Project_Motivation.md).
Using the PyTorch library, I performed transfer learning with VGG19 and ResNet18. Using only CPU calculations I was able to reproduce the binary classification results as reported in the literature (see the Datasets and References below),  in terms of evaluation metrics such as: Accuracy, Precision, Recall, F1 score, and AUC score.

## Datasets
For more details see [Datasets](https://github.com/bcrodrigo/capstone_project/blob/main/references/Dataset_Details.md).

## Directory Structure
```nohighlight
.
├── README.md
├── LICENSE
├── jupyter_notebooks/
├── pytorch_environment.yml
├── references/
├── reports/
└── src/
    ├── data/
    ├── models/
    └── visualization/
```

## Environment Installation
1. Download or clone the project and change to the project directory.
2. Create a virtual environment from the `.yml`  file as follows
```bash
conda env create -n my_new_env -f pytorch_environment.yml
```
3. Download the image datasets from kaggle. Please see the following file for details of the specific [Datasets](https://github.com/bcrodrigo/capstone_project/blob/main/references/Dataset_Details.md) used in this project.
4. See the `jupyter_notebooks` folder for more [details](https://github.com/bcrodrigo/capstone_project/blob/main/jupyter_notebooks/Notebook_Details.md) on the preprocessing steps and the expected directory structure.

## Work In Progress
- Uploading trained models
- Streamlit demonstration of model predictions

## Next Steps
- Implement device selection (CPU, GPU, MPS).
- Implement more scripting and automation
- Explore additional training strategies
	- Multiclass classification (→ need to rebalance dataset)
	- Fine-tuning
	- Hierarchical structure classification
- Experiment with Vision Transformers

## References
1. A. Khan, B. Hassan, S. Khan, R. Ahmed and A. Adnan, *DeepFire: A Novel Dataset and Deep Transfer Learning Benchmark for Forest Fire Detection* Mobile Information System, vol. 2022, pp. 5358359, 2022 [doi](https://doi.org/10.1155/2022/5358359).
2. El-Madafri I, Peña M, Olmedo-Torre N. *The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach.* Forests. 2023; 14(9):1697.  [doi](https://doi.org/10.3390/f14091697)