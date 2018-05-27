# Image to Image Mappings


This repository is based on the first proposed exercise of chapter 3 of the book *Programming Computer Vision with Python* by Jan Erik Solem. In this exercise the idea is to take the image coordinates of a square (or rectangular) object for estimate the transformation that takes the rectangle to a full on frontal view.

## Files

* ``data`` has 4 test images, two from a book and two from a card.

* ``ch03-ex1.py`` has the source code. To run it you need write in the command line: ``python ch03-ex1.py data/image.jpg``, change ``image`` by any image name of the ones in the ``data`` folder. Once you run the above command you need to select with right click the 4 corners of book or card. Keep in mind that the algorithm designed by default put the output image in a vertical orientation.

* ``ch03-ex1-nb.ipynb`` has the documentation of code that be in ``ch03-ex1.py`` where mainly the function that calculates the homography is explained.