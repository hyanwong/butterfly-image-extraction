butterfly-image-extraction
==========================

A collection of Python scripts to extract outlines of butterflies from pictures of pinned specimens, 
with images assumed to be downloaded from the Encyclopedia of Life (http://eol.org).

Images have a "data object ID", such as 26289989, which corresponds to the picture at http://eol.org/data_objects/26289989. They are usually assume to come in two versions: a low-resolution jpg (maximum 560x380 pixels), and a full resolution jpg. For example, for the image 26289989, these are located at http://media.eol.org/content/2013/09/22/03/25762_560_380.jpg and http://media.eol.org/content/2013/09/22/03/25762_orig.jpg

The main file is butterfly_detection.py. You can either place a set of files in a folder and call 

    python butterfly_detection.py image_dir

or if you have a csv file with header ID,URL followed by lines giving the data object ID,URL for a set of images, then you can call the script as

    python butterfly_detection.py list_of_images.csv

If you call the script with no parameters, it looks for a file of this sort called "butterflies.csv". This will save a tiled composite of the stages in the identification process in a folder called "classification" and the final butterfly image in a folder called "butterflies". It may also save the coutour outlines of various shapes.

Another possibility is to run the script called butterfly_comparisons.py which will run the same routine, but simply output the final shapes (if the "save" flag is given) or compare them against a known set of shapes.
