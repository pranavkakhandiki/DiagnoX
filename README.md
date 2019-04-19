# DiagnoX
DiagnoX is an open source project dedicated to diagnosing diseases. The original code is designed and geared towards Aortitis, a rare heart disease characterized by inflammation in the aorta. Whether you are searching for inspiration for a Science Fair project, a new hobby, or you just want to give back to the community, DiagnoX is a great way to start!

To store and compare the CT scans, the python program uses HOG (histogram of oriented gradients) descriptors. They store a histogram of gradients (consisting of x and y derivatives which have direction and magnitude), which is more efficient than storing the entire image because the “useful” data consists of abrupt changes in the derivatives.

For classification, Linear-SVC, an algorithm which establishes a hyperplane between clusters of data, is used. LinearSVC uses the parameters which the HOG descriptor provides to train the program and draw the hyperplane, effectively classifying each image as either having or not having an inflamed aorta.

With an overall accuracy rate of 94%, and a type II (false negative) error rate of only 1.4%, the algorithm proves to be effective.


# HOW TO GET STARTED

1. Gather Data: Obtain images of your preffered disease, preferably over 50 images for the program to work well, and convert then to .png (using XnView)

2. Download Label-img: after downloading, use the following commands to open it (in terminal):
cd [INSERT DIRECTORY]
python3 labelimg.py

3. After storing the images in a directory, open that directory using label-img. Then, 'annotate' each image with the feature you want the program to notice/diagnose. In the case of aortitis, it is the thickened aortic wall. You will likely have to create a 'predifined classes' file before annotating the images. If anything is not working, referece the following link: https://pypi.org/project/labelImg/

4. Run the program in PyCharm or your favorite python IDE! - It's that easy, now all that's left is to tweak it to improve its accuracy for the disease you wish to diagnose.

What do I change? - Some variables to start tweaking are:
From hog.py: orientations, pixelsPerCell, cellsPerBlock
From train.py: thresh, image_resize
From data_loader.py: size, img_slice, img_data

Note: while playing around with values and configurations of the variables listed above may help, some diseases may require changes in the actual algorithm. While this may require more effort, it's definitely worth it!

# License
This project is licensed under the Apache 2.0 license
