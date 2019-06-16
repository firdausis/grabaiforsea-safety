# grabaiforsea-safety
My submission for the [Grab AI for SEA - Safety Challenge](https://www.aiforsea.com/safety).

This consists of 3 parts:
- [Preprocessing](Safety%20Challenge%20-%20Preprocessing.ipynb) - used for data cleansing and feature extraction.
- [Training](Safety%20Challenge%20-%20Training.ipynb) - used for model training.
- [Testing](Safety%20Challenge%20-%20Testing.ipynb) - used for model evaluation.

### Data Cleansing

The following telematics data are removed:
- Data with invalid speed (negative or more than 300 km/h)
- Data with low accuracy (more than 50 meters)
- Trips with unrealistic duration (more than 12 hours)
- Trips with insufficient telematics records (less than 100 seconds)

### Feature Extraction

The following features are extracted to be used in the model:
- Speed-based (max, mean, IQR, max change)
- Acceleration-based (min, max, mean, IQR, max change)
- Gyro-based (min, max, mean, IQR, max change)
- Trip duration
- Total distance
- Total rotation

### Model Training

XGB algorithm is used since usually it outperfoms other traditional methods.

### Instructions

- Training

The preprocessing and model training step are already done with the output file [dataset-ready.csv](dataset-ready.csv) and [xgb.model](xgb.model). If you want to re-train, extract the [training dataset](https://s3-ap-southeast-1.amazonaws.com/grab-aiforsea-dataset/safety.zip) and run [Safety Challenge - Preprocessing.ipynb](Safety%20Challenge%20-%20Preprocessing.ipynb) and [Safety Challenge - Training.ipynb](Safety%20Challenge%20-%20Training.ipynb).

- Evaluation

Firstly you need to run [Safety Challenge - Preprocessing.ipynb](Safety%20Challenge%20-%20Preprocessing.ipynb) with modifying the input dataset and output feature file config. Then you need to run [Safety Challenge - Testing.ipynb](Safety%20Challenge%20-%20Testing.ipynb) with modifying the model file and input feature file config.
