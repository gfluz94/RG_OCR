# RG_OCR

This repository is intended to perform Optical Character Recognition in a Brazilian ID document.

In this project, it is assumed that the input image may not be in the best perspective in order for this task to be performed. Therefore, some steps must be taken to preprocess the image, before applying OCR.

### Dewarping

1. Image is read in gray-scale;
2. Gaussian Blur is performed to remove any available noise;
3. Adaptative Threshold is applied to blurred image;
4. We find the contour whose area is the greatest, since it represents the document frame;
5. With the contour found in the last step, we create a mask with the area represented by the frame;
6. Using this mask, we are able to find the four corners of the ID document in the original image;
7. Therefore, we apply **dewarping** and transform our perspective, so the four corners of the document are the same as the image.

### OCR

Since we have an image with better resolution and we know the document's template, we apply Optical Character Recognition to it by using **pytesseract** API.

### Web Application

We also have a web application developed with **Flask**, so the user can upload an image of the document and then the text information contained within is displayed on the screen.
