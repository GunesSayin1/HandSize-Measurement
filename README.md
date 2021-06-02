## Basic Overview

This application measures the different parameters of the hand by using Image Processing and Computer Vision technologies. 

Example output can be seen below:

![hand](https://user-images.githubusercontent.com/62821891/120560117-7cefc880-c402-11eb-94c3-8ec772205921.png)


### Reproducing Results

1. Install the requirements.txt using pip

   ```
   pip install -r requirements.txt
   ```

2. Open the CVIP_Main_assignment.py, create a new class instance and call the method plot_table.

   ```python
   hand = Hand_measurement("Image (9).jpg", "left",19,255)
   newhand.plot_table()
   ```

3. Done !

### References:

- https://pierfrancesco-soffritti.medium.com/handy-hands-detection-with-opencv-ac6e9fb3cec1
- https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08

- https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/

- https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

- https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python





