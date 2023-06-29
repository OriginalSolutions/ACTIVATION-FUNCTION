# ACTIVATION FUNCTION
ACTIVATION FUNCTION TESTER FOR AI
***

The goal of this project is to promote, the building and testing of new or custom activation features.
The solution is based on a class that defines a neural network placed on the website: https://realpython.com/python-ai-neural-network/   


Selected methods have been removed from the class and new ones have been defined. The most important changes include:

Methods removed:
* "Sigmoid" activation function
* derivative function of the "Sigmoid" function

Added methods:
* "Mish" activation function
* derivative function of the "Mish" function
* six "hidden" (deep) layers of the network
* neurons - average of the results of neurons transferred to the activation function

Added class attributes:
* "random seed" with a high parameter - in order to eliminate randomly generated linear forecasts. 

Moreover, in the main script to improve prediction performance, loop have been defined that repeat training.


***


The inspiration for this project are the conclusions drawn from testing our own AI projects based on the use of external libraries.

Tests have been performed on the maximum number of:
* neurons - 1024
* layers - 10 
* epochs - 400

Tests show that the most important ingredient for AI aimed at prediction is a properly chosen activation function.

Whereas: incorrectly selected activation function, makes:
the number of neurons, network layers  and of learning epochs is of secondary importance.

