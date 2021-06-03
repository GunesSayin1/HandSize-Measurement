## Hand Size Measurement

## Basic Overview

This application measures the different parameters of the hand by using Image Processing and Computer Vision technologies. 

Example output can be seen below:

![download (4)](https://user-images.githubusercontent.com/62821891/120564514-5eda9600-c40b-11eb-92d9-143321406b9a.png)

###### Directory Structure

```
Hand_Measurement
├── handscans                  #Folder containing handscan csv and images
│   └── handscans.csv
│   └── Image .jpg
│   └── ...
│ 
├── CVIP_Main_Assignment.ipynb #Notebook file containing application code and examples
├── CVIP_Main_Assignment.pdf   #PDF Version of notebook file with same name
├── CVIP_Main_assignment.py	   #Python script version of the application
├── Example Results .pdf       #Some example results without much code in PDF format
├── Step-by-step .ipynb        #Notebook file step by step executing the methods in main class and explaining steps taken.
├── Step-by-step .pdf      	   #PDF Version of notebook file with same name
├── requirements.txt           #pip requrirements
├── README.md
```



### Reproducing Results

1. Install the requirements.txt using pip

   ```
   pip install -r requirements.txt
   ```

   ##### Single Image Usage

   1. Open the CVIP_Main_assignment.py, create a new class instance and call the method plot_table.

      ```python
      hand = Hand_measurement("handscans/Image (9).jpg", "left",19,255)
      newhand.plot_table()
      ```

   
   ##### Multiple Image Usage with CSV File
   
   1. Open the CVIP_Main_assignment.ipynb, assign a variable to the csv filtering function.
   
      ```python
      good_quality=quality("handscans/handscans.csv",False)
      ```
   
   2. Using for loop create class instances for each file name and side of the hand, and plot them using plot_table method.
   
      ```python
      for image,side in zip(good_quality["path"],good_quality["hand"]):
          newhand = Hand_measurement(image, side)
          plt.figure(figsize=(9,9))
          newhand.plot_table()
      ```
   
   ##### Writing results to CSV file
   
   1. Open the CVIP_Main_assignment.ipynb, assign a variable to the csv filtering function.
   
      ```python
      good_quality=quality("handscans/handscans.csv",False)
      ```
   
   2. Using for loop create class instances for each file name and side of the hand, and write them to csv name given in argument.
   
      ```python
      for image,side in zip(good_quality["path"],good_quality["hand"]):
          newhand = Hand_measurement(image, side)
          newhand.csv_measurements("measurements.csv")
      ```
   
   [^Note]: To keep the file size of the project smaller, only good and reasonable quality images are included in the project folder.
   
   

### References:

- https://pierfrancesco-soffritti.medium.com/handy-hands-detection-with-opencv-ac6e9fb3cec1
- https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08

- https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/

- https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

- https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python





