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
### Machine Learning Algorithms
#### 1) 

### Data Augmentation
