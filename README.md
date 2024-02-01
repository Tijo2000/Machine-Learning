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
## Learning Rate
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/823e2f21-419e-42fd-b04c-3671b400bcf8)
Imagine you're training a dog to sit. With each iteration (treat you give), the dog learns and gets closer to mastering the "sit" command. But how big of a jump should the dog make in its understanding with each treat? That's where the learning rate comes in.

In machine learning, the learning rate controls how much the model's parameters are adjusted after each training iteration. It determines how quickly the model "learns" from the data and updates its predictions.

Here's a breakdown:

High learning rate:

Like giving the dog a big leap of treats, it can lead to fast changes in the model.
Good for exploring the solution space and escaping local minima (shallow dips in the error landscape).
But be careful! It can also cause the model to overshoot the optimal solution and become unstable, like the dog jumping and missing the "sit" completely.
Low learning rate:

Like giving small nudges with treats, it leads to gradual changes in the model.
More likely to converge to a stable solution (like the dog eventually learning to sit properly).
But take too long! It can get stuck in local minima or miss deeper valleys (better solutions) in the error landscape.
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



