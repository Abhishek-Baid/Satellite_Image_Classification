# Satellite Image Classification

## Objective
The public and continuous access to image data from satellites for Earth observation is availabe from different agencies like NASA, EU, ISRO etc. However to utilize this data for semantics like Land Use and Land Cover Classification , we need automatic labelling describing the physical land type , for ex: residential, forest etc. This project aims to take one such dataset from EU Sentinel 2 satellite and use different machine learning and deep learning techniques for classification to assess use of such techniques on this data and compare the results.

## Dataset
The dataset taken contains 27000 images from EU Sentinel 2 satellite. Sentinel 2 consists of 2 satellites. First came Sentinel 2A which was launched in 2015. Next came Sentinel 2b in 2017. Two additional satellites (Sentinel 2C and 2D) are planned to launch in 2024. This will make a total of four Sentinel-2 satellites. Overall, these 2 additional satellites will cut the revisit time in half.
The images are available in 12 bands with different resolutions. For our observation we are going to work with 64x64 RGB band (B2, B3, B4) images with 10m resolution (i.e. each pixel shows an area of 10m). 

![image](https://github.com/Abhishek-Baid/Satellite_Image_Classification/assets/33655961/59baa96e-3ef4-4be4-9a08-16b21aabe3ec)


## Project Approach 
The project aims to compare different ML/DL models and techniques applied on EuroSAT dataset of 27000 images and observe the results of classification among 10 classes. 

### A. Data Processing and Data Augmentation
For the machine learning algorithms , the EuroSAT RGB dataset is loaded using CV2 library and converted to numpy arrays. Further to improve performance, data augmentation was done on images with 1:1 ratio , doubling the dataset. Due to memory contraints, further augmentation was not possible and these datasets were used with different classification algorithms with different properties.
For the deep learning solution, tensorflow's ImageDataGenerator , flow from directory and flow from dataframe methods were used for better results using image augmentation. 

### B. Model Summary 
### Machine Learning Algorithms : Models Applied and Results

|S.No.|Model|Parametes|Accuracy|Precision|Recall|F1-Score|
|---|---|---|---|---|---|---|
|1 |	SGD |	Default |	0.237|	0.282|	0.26|	0.208|
|2 |	SGD |	max_iter=1500, tol=0.0001, early_stopping|	0.415|	0.403|	0.409|	0.401|
|3 |	SGD |	max_iter=1500, tol=0.0001|	0.395|	0.465|	0.384|	0.392|
|4 |	KNN |	n_neighbours=10| 0.34|	0.29|	0.34|	0.24|
|5 |	KNN |	n_neighbours=7|	0.338|	0.49|	0.34|	0.27|
|6 |	KNN |	n_neighbours=15, algo='kd_tree'|	0.337|	0.49|	0.3|	0.24|
|7 |	LogisticRegressionCV |	multi_class='multinomial', penalty='l2', solver='lbfgs', Cs=[0.01, 0.1, 1, 10], random_state=83|	0.41||||			
|8 |	Linear SVM Classifier |	max_iter=100 |	0.245|	0.296|	0.246|	0.248|
|9 |	SVM poly kernel |	SVC(kernel="poly", degree=3, coef0=1, verbose=True, C=5  )|	0.63|	0.636|	0.625|	0.624|
|10 |	SVM poly kernel |	SVC(kernel="poly", degree=5, coef0=1, verbose=True, C=1  )|	0.64|	0.644|	0.634|	0.635|
|11 |	Bagging Classifer |	500 decision trees|	0.6|	0.619|	0.626|	0.614|
|12 |	RandomForestClassifier |	number of trees =500,  max_leaf_nodes allowed =15 with criterion as 'entropy'|	0.49|	0.498|	0.426|	0.402|
|13 |	RandomForestClassifier |	Number of trees = 1000, criterion='gini' with no threshold on leaf nodes| 	0.698|	0.686|	0.691|	0.682|
|14 |	RandomForestClassifier |	Number of trees = 1300, criterion='gini' with no threshold on leaf nodes|	0.699|	0.688|	0.692|	0.683|

### Neural Net Based Models

|S.No.|	Model|	Approach|	Validation Loss|	Test Accuracy|	Precision|	Recall|	F1-Score|
|---|---|---|---|---|---|---|---|
|1 | Simple Convolution|	3 convolution layers with dropout and 3 dense layers.  Optimizer=Nadam(learning_rate=0.000005), loss='sparse_categorical_crossentropy' , metrics=['accuracy'] | ||||| 					
|2 |	Resnet34 (Self Implementation)|	optimizer=Nadam(learning_rate=0.000005,clipnorm=1.0), loss='sparse_categorical_crossentropy' , metrics=['accuracy']|	0.356|	0.881|	0.875|	0.874|	0.874|
|3	| Resnet50(Transfer Learning) |	optimizer=Nadam(learning_rate=0.000001,clipnorm=1.0), loss='sparse_categorical_crossentropy' , metrics=['accuracy'] | 	0.234|	0.928|	0.927|	0.925|	0.926|
|4 |	Inceptionv3(Transfer Learning)| optimizer=Nadam(learning_rate=0.000001,clipnorm=1.0), loss='sparse_categorical_crossentropy' , metrics=['accuracy'] | 	0.231|	0.937|	0.936|	0.935|	0.935|
|5 |	InceptionResnetv2(Transfer Learning)|	optimizer=Nadam(learning_rate=0.000001,clipnorm=1.0), loss='sparse_categorical_crossentropy' , metrics=['accuracy']| 	0.173|	0.938|	0.938|	0.936|	0.937|
|6 |	VGG19|	optimizer=Nadam(learning_rate=0.000001,clipnorm=1.0), loss='sparse_categorical_crossentropy' , metrics=['accuracy'] |	0.15|	0.838	|0.85|	0.83|	0.84|





### Data Augmentation
