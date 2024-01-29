# Day 1
## Supervised Learning
 Machine learning is a branch of artificial intelligence (AI) and computer science that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. Machine learning algorithms are used in a wide variety of applications

Learns from being given **right answers**.
Supervised machine learning is based on the basis of labeled data.First the data is fed to the model with both input and output and later on test data is given to make prediction by model. some algorithm used in supervised learning with their uses are :

Regression : House price prediction </br>
Classification : Breast cancer detection.

# Day 2
## Unupervised Learning
Learns by finding pattern in unlabelled data. Unsupervised learning is different from supervised learning as it is not provided with labelled data.The algorithm work by finding pattern in data.
some algorithm used in unsupevised learning with it uses are:

Clustering : Grouping similar data points together e.g: grouping of customer , grouping of news,DNA microarray.</br>
Anomlay detection: Finding unusal data points e.g: fraud detection , quality check.</br>
Dimensionality reduction : Compress data using feweer numbers e.g : Image processing.</br>

# Day 3
## Univariate Linear Regression

Imagine you have a bunch of dots scattered on a graph, with some dots higher or lower than others. Linear regression wants to draw the best-fitting straight line through these dots. This line captures the overall trend between two variables, even though the dots won't perfectly align with it.

Think of it like predicting the weight of an apple based on its diameter. You measure many apples, take their diameters and weights, and plot them on a graph. Linear regression finds the "best guess" line that goes through the middle of these dots, so you can use it to predict the weight of a new apple just by knowing its diameter.

Here's the basic mathematical intuition:</br>
Equation of a line: y = mx + b, where m is the slope and b is the y-intercept.</br>
Goal: Minimize the distance between the actual values of your data points (y) and the values predicted by the line (mx + b).
Distance measure: We often use the sum of squared differences between actual and predicted values. This ensures larger errors have a bigger impact on the line's position.
Finding the best line: Techniques like gradient descent adjust m and b iteratively to minimize the sum of squared differences. Think of it like rolling a marble down a hill with valleys representing low error and peaks representing high error. You want the marble to settle in the deepest valley, which is the best-fitting line.

## Cost Function
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/f143c893-239c-4869-a575-6458a3c30ff4)
A cost function takes the predicted values from your model and compares them to the actual values in your data.
It measures the difference between these values and quantifies it into a single number, representing the "cost" of those predictions.
The lower the cost, the better your model's predictions align with reality.
Why it's important:

The cost function acts as a feedback mechanism for your model.
By minimizing the cost function, your model learns and adjusts its parameters to make better predictions in the future.
Different algorithms use different cost functions depending on the type of problem they're solving.
Examples:

Mean squared error (MSE): Commonly used for regression problems, it calculates the average squared difference between predictions and actual values.
Cross-entropy: Used in classification problems, it measures the information gain or loss from your model's predictions.


