# Content-Based-Image-Retrieval
In this project, you are asked to identify the 5 most similar pictures to a picture by evaluating the color similarities of the pictures.
Process Steps: It consists of 2 stages.

1. Preparing the pictures in the database: Perform the following operations only once for all the pictures in the training database. Save your results to use for test pictures.
Calculation of color histograms: Color histograms of images will be used to measure color similarity. All images in the image database
• (R,G,B) histograms (you will have a total of 3 histograms for R,G and B separately)
• H histogram in HSV space
Calculate only once. Normalize the results to the [0,1] range by dividing the results by the total number of pixels.

2. Measuring System Success with Sample Test Pictures: Perform the following steps during the test phase.
a. Calculate the histograms of the test image (R,G,B) and (H) for each test image as you did for the training samples.
b. While measuring the similarities of the pictures, calculate the distance of the given test picture to all the training pictures separately for the (R, G, B) space and (H) using the Euclidean Distance method. For both spaces, find the 5 images in which the test image is most similar. 
