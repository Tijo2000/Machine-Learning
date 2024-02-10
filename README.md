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

# Day 4
## Gradient Descent and Its Intuition
### **Gradient descent is an algorithm for finding values of parameters w and b that minimize the cost function J.**
Imagine you're lost in a hilly landscape, searching for the lowest valley (minimum point). Gradient descent is like an algorithm that helps you find that valley by taking small steps downhill. Here's the basic idea:

Start on a hill: You start at some random point on the hill (initial parameters for your model).
Feel the slope: You calculate the "slope" of the hill at your current position (the gradient). This slope tells you how steep the hill is and in which direction you should go to descend.
Take a small step: You take a small step in the opposite direction of the slope (adjusting your model parameters a little) here w and b
Repeat: You keep repeating steps 2 and 3, feeling the slope at each new position and taking small steps until you reach a point where the slope is zero (you've found a minimum!).
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/ef563786-1a81-45b8-b9b5-4da1dd1cc5e2)
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/8dba1866-9f80-402d-8c1e-1197ae840df1)


Mathematical intuition:

The hill you're exploring represents the cost function, which measures how "wrong" your model's predictions are. Lower values mean you're closer to the optimal solution.
The slope (gradient) is calculated using derivatives, which tell you how much the cost function changes with respect to small changes in your model parameters.
The step size determines how large or small your adjustments are. A large step might get you out of a local minimum, while a small step might take too long to reach the global minimum.
<img width="538" alt="image" src="https://github.com/Tijo2000/Machine-Learning/assets/56153083/2aad1443-2318-4cae-8c23-5e72f1bcffc6">

Here are some key points to remember:

Gradient descent doesn't guarantee finding the absolute best solution (global minimum), but it often gets you close.,</br>
You need to choose a good learning rate (step size) for optimal performance.



# Day 4
## Learning Rate(Alpha)
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/823e2f21-419e-42fd-b04c-3671b400bcf8)
Imagine you're training a dog to sit. With each iteration (treat you give), the dog learns and gets closer to mastering the "sit" command. But how big of a jump should the dog make in its understanding with each treat? That's where the learning rate comes in.

In machine learning, the learning rate controls how much the model's parameters are adjusted after each training iteration. It determines how quickly the model "learns" from the data and updates its predictions.

High learning rate:

If learning rate is too large gradient descent may overshoot and never reach minimum i.e fail to converge,diverge.</br>

Low learning rate:

If learning rate is too small gradient descent may be too slow and take much time.

Choosing the right learning rate:

It's a balancing act! You want the model to learn quickly but not miss the best solution.
Often adjusted through trial and error or adaptive optimization techniques.
Can be different for different parameters or even change during training.
Key points:

Learning rate is crucial for model performance and stability.
Too high can lead to instability and overfitting.
Too low can lead to slow convergence and underfitting.
Choosing the right rate depends on the problem and model.

# Day 4
## Algorithms using Gradient Descent and Its types
Algorithms Using Learning Rate and Gradient Descent and Different Gradient Descent Types
Many machine learning algorithms leverage learning rate and gradient descent for optimization. Here are some prominent examples:

Linear Regression: Gradient descent with learning rate adjustments is the core optimization technique for finding the line that best fits your data in linear regression.

Logistic Regression: Similar to linear regression, this algorithm utilizes gradient descent and learning rate to optimize the model parameters for classifying data points.

Support Vector Machines (SVM): SVMs employ gradient descent variations like stochastic gradient descent to find the optimal hyperplane separating different classes in data.

Neural Networks: Training neural networks relies heavily on gradient descent with learning rate adjustments. Backpropagation, a key training algorithm, uses it to propagate errors backward and update network weights.

Decision Trees: While not directly using gradient descent, decision trees can benefit from learning rate-like parameters controlling tree growth complexity.

Gradient Boosting: This ensemble method aggregates weak learners trained with gradient descent and learning rate tuning for improved predictive power.

Deep Learning Algorithms: Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and other deep learning architectures heavily rely on complex variations of gradient descent with adaptive learning rate adjustments for efficient optimization.

Different Types of Gradient Descent
Beyond the basic concept, various gradient descent approaches cater to specific optimization needs:

Batch Gradient Descent: Updates model parameters after processing all training data. Suitable for small datasets but can be computationally expensive for larger ones.

Stochastic Gradient Descent (SGD): Updates parameters for each individual training example, making it faster for large datasets but potentially noisy.

Mini-batch Gradient Descent: Processes data in batches and updates parameters after each batch, balancing efficiency and stability between batch and SGD.

Momentum: Incorporates inertia to help escape local minima and accelerate convergence by considering the direction of previous updates.

Adam: An adaptive learning rate optimization algorithm adjusting the learning rate for each parameter based on their historical updates and gradients.

RMSprop: Another adaptive learning rate method using squared gradients for more stable updates compared to standard SGD.

Adagrad: Also adaptive, but can suffer from diminishing learning rates for parameters with frequent updates.

# Day 5
## Multiple Linear regression
Multiple linear regression in machine learning model that uses multiple variables called as features to predicts the output.
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/1cf29bcf-8c6f-4de1-bf57-4295197f4d3c)

## Vectorization for faster calculation 
In muliple linear regression calculation is done using vectorization as it perform all calculation simultaneously and parallely and speed up the arithmetic operations.
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/e5b96258-f2c1-40a3-804c-0b746d7593bb)

# Day 6
## Feature Scaling
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/fc35f513-1ba5-4ef3-a3ae-55a2b0674ba7)

Imagine you're training a dog to sit. Different dogs come in different sizes, so if you reward a tiny Chihuahua with the same size treat as a giant Great Dane, they'll react very differently. This is similar to how features work in machine learning algorithms. Some features might have large values, while others might be tiny. If you don't adjust for this, the algorithm might prioritize the features with bigger values, even if they're not actually the most important.

Feature scaling helps by putting all features on the same "playing field." It's like resizing all the dogs to the same size, so the treat reward makes sense for everyone. There are different ways to achieve this, some more common than others:

1. Min-Max Scaling: This squishes all values between 0 and 1. Imagine shrinking all the dogs down so the smallest becomes 1 foot tall and the largest becomes 5 feet tall.

2. Standardization: This centers all values around 0 with a standard deviation of 1. Think of moving all the dogs to the same starting line, where the average dog is in the middle and everyone else is spread out evenly around them.

3. Normalization: This brings all values within a specific range, like -1 to 1. Like putting all the dogs in a specific size cage, regardless of their original size.

Which type of scaling to use depends on your data and algorithm. Some algorithms are sensitive to the scale of features, while others don't care. It's best to experiment and see what works best for your specific case.

Here are some key points to remember:

Feature scaling makes your data more manageable for machine learning algorithms.
Different types of scaling exist, each with its own pros and cons.
Choosing the right scaling method depends on your data and algorithm.



### Difference between Normalization and Standardization 
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/31dd43bc-c31f-4de8-9451-8fc944f774c5)

Normalization and standardization are both techniques used in feature scaling for machine learning, but they differ in how they achieve the same goal of putting all features on a similar scale. Here's a breakdown of their key differences:

Scale:

Normalization: Normalizes features to fall within a specific predefined range, often between 0 and 1 or -1 and 1. This makes features easy to interpret in terms of the chosen range.
Standardization: Standardizes features to have a mean of 0 and a standard deviation of 1. This doesn't place features within a specific range but ensures they have equal importance in the model's calculations.
Sensitivity to outliers:

Normalization: Can be significantly affected by outliers, as it uses the overall minimum and maximum values for scaling. Outliers can skew the range and distort the representation of other features.
Standardization: Less sensitive to outliers because it uses the mean and standard deviation, which are calculated based on all data points and are less influenced by extreme values.
Distance-based algorithms:

Normalization: More suitable for algorithms that rely on distances between data points, like k-Nearest Neighbors, because features have a common range.
Standardization: More suitable for algorithms sensitive to feature scale, like linear regression and gradient descent, as feature values have equal weight on the model's output.
Choosing the right technique:

Consider the algorithm you're using and its sensitivity to feature scale.
Think about the presence of outliers in your data and their potential impact on scaling.
If interpretability of feature values within a specific range is important, choose normalization.
If you want all features to have equal weight and are less concerned about interpreting specific values, choose standardization.

## Choosing correct learning rate
First we make sure gradient descent is decreasing over the iteration by looking at learning curve if it is working properly we choose correct learning rate by starting with smaller learning rate and increase it gradually.




