# Image Classification of Forest Fires with Deep Neural Networks
In this project I use pre-trained Computer Vision Models to perform image classification of Forest Fires. For more details about the project motivation see [Project Motivation](https://github.com/bcrodrigo/capstone_project/blob/main/reports/Project_Motivation.md).

Using the PyTorch library, I performed transfer learning with VGG19 and ResNet18. With only CPU calculations I was able to reproduce the binary classification results as reported in the literature in references [^1] and [^2] listed below. 


**Table 1:** Binary classification results for VGG19 and ResNet18 trained with the [DeepFire Dataset](https://www.kaggle.com/datasets/alik05/forest-fire-dataset)

|  Metric   | VGG19 reported in [^1] | VGG19  | ResNet18 |
| :-------: | :--------------------: | :----: | :------: |
| Accuracy  |         0.9500         | 0.9763 |  0.9789  |
| Precision |         0.9572         | 0.9641 |  0.9643  |
|  Recall   |         0.9421         | 0.9895 |  0.9947  |
| F1 score  |         0.9496         | 0.9766 |  0.9793  |
| AUC score |         0.9889         | 0.9763 |  0.9789  |

**Table 2:** Binary classification results for VGG19 and ResNet18 trained in the [WildFire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset/data)

|  Metric   | MobileNetV3 reported in [^2] | VGG19  | ResNet18 |
| :-------: | :--------------------------: | :----: | :------: |
| Accuracy  |            0.8405            | 0.8512 |  0.8610  |
| Precision |            0.8322            | 0.8267 |  0.8400  |
|  Recall   |            0.7799            | 0.7799 |  0.7925  |
| F1 score  |            0.8049            | 0.8026 |  0.8155  |
| AUC score |            0.8397            | 0.8381 |  0.8484  |

## Datasets
For more details see [Datasets](https://github.com/bcrodrigo/capstone_project/blob/main/references/Dataset_Details.md).

## Online Streamlit App
Follow [this link](https://image-classification-forest-fires.streamlit.app/) to see an online demonstration of the predictions of the models.

## Directory Structure
```bash
.
├── LICENSE
├── README.md
├── jupyter_notebooks/
├── model_demo/
├── pytorch_environment.yml
├── references/
├── reports/
├── requirements.txt
├── src/
│   ├── data/
│   ├── models/
│   └── visualization/
└── streamlit_online_app.py
```

## Environment Installation
1. Download or clone the project and change to the project directory
2. Create a virtual environment from the `pytorch_environment.yml`  file as follows
```bash
conda env create -n my_new_env -f pytorch_environment.yml
```
**NOTE:** The `requirements.txt` file is intended to list the dependencies of `streamlit_online_app.py`

3. Download the image datasets from kaggle. Please see the following file for details of the specific [Datasets](https://github.com/bcrodrigo/capstone_project/blob/main/references/Dataset_Details.md) used in this project.
4. See the `jupyter_notebooks` folder for more [details](https://github.com/bcrodrigo/capstone_project/blob/main/jupyter_notebooks/Notebook_Details.md) on the preprocessing steps and the expected directory structure.

## Work In Progress
- [ ] Uploading trained models
- [ ] Implement device selection (CPU, GPU, MPS).
- [ ] Implement more scripting and automation

## Next Steps
- Explore additional training strategies
	- Multiclass classification (→ need to rebalance/augment WildFire dataset)
	- Fine-tuning
	- Hierarchical structure classification
- Experiment with Vision Transformers

## References
[^1]: A. Khan, B. Hassan, S. Khan, R. Ahmed and A. Adnan, *DeepFire: A Novel Dataset and Deep Transfer Learning Benchmark for Forest Fire Detection* Mobile Information System, vol. 2022, pp. 5358359, 2022 [doi](https://doi.org/10.1155/2022/5358359).
[^2]: El-Madafri I, Peña M, Olmedo-Torre N. *The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach.* Forests. 2023; 14(9):1697.  [doi](https://doi.org/10.3390/f14091697)