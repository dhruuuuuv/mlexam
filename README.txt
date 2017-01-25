submission bvc981, Dhruv Chauhan

DETAILS
-------

The programs are written using python3, so assuming that python3 is installed on your machine, the whole program can be run by executing:

    python exam.py

after unzipping all the files. This results in a printed section of each part - the graphing library halts the progress of the program until the window is shut, so in order to continue, you may have to shut the resulting graph. Equally, the graphs may be commented out from the code by finding the relevant 'plt.show()' function.

DEPENDENCIES
-------------

The main 2 python programs use the following dependencies:

    import csv
    import math
    import operator

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sklearn import svm, preprocessing, linear_model, neighbors, decomposition
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import zero_one_loss
    from sklearn.cluster import KMeans

where matplotlib, plotly, pandas, sklearn (scikit-learn) and numpy may need to be installed separately. This can be done using pip or conda.

FILE STRUCTURE
--------------

The program is split into the main 'exam.py' file that imports and calls the 'galaxies.py' and the 'weeds.py' python programs, that each hold and run the code for their respective class. Inside those files, other python files may be imported and called, such as those from previous assignments.
