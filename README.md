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

# Day 7
## Feature Engineering
### Feature engineering
Feature engineering means designing newfeatures by transforming or combining original features which maybe very important in prediciting the output.
for e.g: we have to predict the price of swimming pool and we have length breadth and height of swimming pool as features now we can used feature engineering to create our new feature which is volume which is very important in predicting the price of swimming pool.

### Polynomial regression
Polynomial Regression is a regression algorithm that models the relationship between a dependent(y) and independent variable(x) as nth degree polynomial. The Polynomial Regression equation is given below:
y= b0+b1x1+ b2x12+ b2x13+...... bnx1n

# Day 8
## Logistic Regression
Logistic regression is a classification technique used to predict the probability of an event occurring, typically belonging to two categories (e.g., yes/no, pass/fail, churn/not churn). Unlike linear regression, which predicts continuous values, logistic regression outputs a value between 0 and 1, representing the likelihood of belonging to a specific category.
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/aa926e69-cc27-4fda-8c72-4e095fb6a1e3)
Model: It represents the relationship between independent variables (features) and the dependent variable (categorical).
Sigmoid function: This function transforms the linear combination of features into a probability between 0 and 1.
Prediction: Based on the probability, a threshold (usually 0.5) is used to classify the observation into a category.

### Sigmoid Function
The sigmoid function is a mathematical function that maps any input value to a value between 0 and 1. It is commonly used in logistic regression to model the probability of a binary outcome. The sigmoid function has an S-shaped curve and is defined as follows:

σ(z) = 1 / (1 + e^(-z))

where z is the input value to the function. The output of the sigmoid function, σ(z), is a value between 0 and 1, with a midpoint at z=0.

The sigmoid function has several important properties that make it useful in logistic regression. First, it is always positive and ranges between 0 and 1, which makes it suitable for modeling probabilities. Second, it is differentiable, which means that it can be used in optimization algorithms such as gradient descent. Finally, it has a simple derivative that can be expressed in terms of the function itself:

d/dz σ(z) = σ(z) * (1 - σ(z))

This derivative is used in logistic regression to update the model coefficients during the optimization process.
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/8a6239ba-778f-4891-81f4-6a9dd4e7effd)

Decision Boundary in Logistic Regression:
In logistic regression, the decision boundary refers to the hypothetical line that separates observations belonging to different classes. It essentially divides the feature space into regions where one class is more likely than the other.

The decision boundary acts as a threshold based on this probability. Typically, a threshold of 0.5 is used:
1. If the predicted probability is greater than 0.5, the observation is classified into the positive class (e.g., churned customer).
2. If the predicted probability is less than or equal to 0.5, the observation belongs to the negative class (e.g., loyal customer).

# Day 9
## Gradient Descent for logistic regression
Logistic Regression Ŷi is a nonlinear function(Ŷ=1​/1+ e-z), if we put this in the above MSE equation it will give a non-convex function as shown:

![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/957a10a7-012d-4598-9641-4b0d32f1eca4)

1. When we try to optimize values using gradient descent it will create complications to find global minima.
2. Another reason is in classification problems, we have target values like 0/1, So (Ŷ-Y)2 will always be in between 0-1 which can make it very difficult to keep track of the errors and it is difficult to store high precision floating numbers.

   ![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/bc41bf89-f8dc-43ad-b7ee-ffa473b01010)
   Gradient Descent in Logistic Regression is an iterative optimisation algorithm used to find the local minimum of a function. It works by tweaking parameters w and b iteratively to minimize a cost function by taking steps proportional to the negative of the gradient at the current point.
Gradient descent in logistic regression looks similar to gradient descent in linear regression but it has different value for function.


# Day 10
## Overfitting and Underfitting
<img width="509" alt="image" src="https://github.com/Tijo2000/Machine-Learning/assets/56153083/81e1826d-a9f5-401f-a148-c1c743a54e3a">

Imagine you're trying to fit a hat on your head. The goal is to find a hat that fits just right, not too loose and not too tight.

Underfitting is like having a hat that's way too large. It sits loosely on your head, doesn't cover your ears properly, and might even blow off in the wind. This happens when a machine learning model is too simple and can't capture the important patterns in the data. It makes bad predictions because it hasn't "learned" enough from the data.

Overfitting is like having a hat that's way too small. It squeezes your head uncomfortably and might even block your vision. This happens when a model is too complex and memorizes the specific details of the training data, including noise and irrelevant information. It performs well on the training data but makes poor predictions on new data because it's focused on those irrelevant details instead of the underlying patterns.

Here's an analogy:

Underfitting: You try to learn a language by memorizing just a few basic words and phrases. You can't understand complex sentences or have meaningful conversations.
Overfitting: You spend so much time memorizing a specific speech that you can't adapt it to different situations or have original conversations.
Finding the right fit:

The best outcome is a well-fitting hat, comfortable and covering your head properly. This is like having a machine learning model that balances complexity and simplicity. It captures the important patterns in the data without memorizing irrelevant details, leading to accurate predictions on both training and new data.

### Underfitting
It is a situtation when the training set doesnot fit well. It happen when data has high bias.

### Overfitting
It is a situation when the training set fit extremely well . It is also known as data with high variance.

Addressing overfitting
1. Collecting more training example
2. Select features include/exclude (feature selection)
3. Reduce the size of parameters i.e "Regularization".
4. overfitting

# Day 11
## Neural Networks
Neural network is an computer algorithms that try to mimic the brain.neural network is made of a input layer that take input data and hidden layer does all the computation and output layer displays the output.
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/8fcac782-b839-43bc-a877-52840c7fbeb8)


### why use neural networks compared to traditional ML algos?
Choosing between neural networks and traditional machine learning algorithms depends on the specific problem you're trying to solve and the data you have available. Here's a breakdown of their strengths and weaknesses to help you decide:

Traditional ML Algorithms:
Strengths:</br>
Interpretability: They often have clear rules and decision boundaries, making it easier to understand how they make predictions.
Efficiency: They can be computationally efficient, especially for smaller datasets.
Stability: They tend to be less prone to overfitting and can perform well even with limited data.

Weaknesses:</br>
Limited complexity: They may struggle with complex, non-linear relationships in data.
Feature engineering: Often require manual feature engineering, which can be time-consuming and domain-specific.
Neural Networks:


Strengths:</br>
High accuracy: They can achieve high accuracy on complex tasks like image recognition, natural language processing, and time series forecasting.
Automatic feature learning: They can automatically learn features from data, eliminating the need for manual feature engineering.
Flexibility: They can handle a wide range of data types and tasks.

Weaknesses:</br>
Interpretability: It can be difficult to understand how they make predictions, making debugging and explaining outputs challenging.
Computational cost: They can be computationally expensive to train, especially for large datasets and complex architectures.
Data requirements: They typically require more data than traditional ML algorithms to achieve good performance.
Here are some specific situations where you might choose one over the other:

Use traditional ML algorithms:
When interpretability is crucial (e.g., in healthcare or finance).
When dealing with small datasets or limited computational resources.
When the problem is relatively simple and well-defined.
Use neural networks:
When dealing with complex data and tasks where high accuracy is needed.
When data is abundant and computational resources are available.
When feature engineering is difficult or impractical.
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/c788f4fe-91fd-491c-8ebd-592e1d99633d)

# Day 12
## Forward Propogation In Neural Networks
Forward propagation in neural networks refers to the process of computing the activation of each neuron in the network layer-by-layer, starting from the input layer and culminating in the output layer.

Key Characteristics:

Feedforward nature: Information flows only in one direction (from input to output).
Parameter dependence: The output depends on the network's weights and biases, which are learned through training algorithms.
Non-linearity: Activation functions introduce non-linearity, allowing the network to learn complex relationships in the data.

Forward propagation is a crucial step in neural network training. It paves the way for backpropagation, where the error between the predicted and desired output is propagated back through the network, enabling updates to weights and biases for improved performance.

# Day 13
## Tensorflow Implementation for Neural Networks
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/0b677ea8-a9b9-4faf-91f9-d0c5632f2040)

### Does Logistic Regression have an Activation Function?

Yes, logistic regression does have an activation function, although it's sometimes not explicitly mentioned. The activation function used in logistic regression is the sigmoid function, also known as the logistic function.

Here's why:

Logistic regression predicts the probability of an event occurring, with outputs ranging from 0 (never happens) to 1 (always happens).
To achieve this probabilistic output, the linear combination of weighted inputs from the model needs to be transformed into a value between 0 and 1.
The sigmoid function, with its S-shaped curve, maps any numerical input between negative and positive infinity to a value between 0 and 1.
Therefore, although the term "activation function" might not be explicitly used in logistic regression, the sigmoid function plays a crucial role in converting the model's linear output into a probability, fulfilling the essential function of an activation function.

Key properties of sigmoid function in logistic regression:

Ensures outputs are limited between 0 and 1, representing probabilities.
Introduces non-linearity into the model, allowing it to learn complex relationships between features and the target variable.
Makes the model differentiable, enabling optimization through gradient descent algorithms.
While the sigmoid function is the traditional choice for logistic regression, other activation functions like the softplus or tanh can also be used in certain applications.

# Day 14
## Tensorflow Training Workflow
<img width="527" alt="image" src="https://github.com/Tijo2000/Machine-Learning/assets/56153083/f69974a5-f99d-4432-a102-6e84f97d9d17">

# Day 15
## Activation Functions
Imagine you're throwing a party and need to decide who gets invited. You have a list of guests with different qualities like being fun, energetic, or famous.

Without an activation function:

You simply add up the points for each quality and invite anyone who reaches a certain score. This is like using a linear activation function. It only considers the sum of the qualities, which isn't very flexible.
With an activation function:

You have a "gatekeeper" who considers the sum of qualities but also adds their own judgment. Maybe they prefer funny people even if they aren't famous, or they appreciate energetic people more if the party is small. This "gatekeeper" is like an activation function.
What activation functions do:

They take the information from previous layers (the guest qualities) and transform it based on certain rules (the gatekeeper's judgment).
These rules can be simple (like a threshold: invite only the top 10%) or more complex (like considering combinations of qualities).
Different activation functions have different "personalities" and are suited for different tasks.
Why are they important?

They add non-linearity to the network, allowing it to learn complex relationships between data points.
They act like filters, controlling what information gets passed on to the next layer.
Choosing the right activation function can significantly improve the performance of your neural network.
Thinking of it another way:

Imagine a light switch. A linear activation function would be like a simple on/off switch, while a non-linear function could be like a dimmer switch, allowing for different levels of brightness.
Remember:

Activation functions are like decision-makers within the network, adding flexibility and power to learning.
Experimenting with different types can help you find the best fit for your specific problem.

<img width="530" alt="image" src="https://github.com/Tijo2000/Machine-Learning/assets/56153083/783ba567-7f29-4de7-bb67-7d4d5984a092">

## Choosing the right activation function
Choosing the right activation function for your output layer depends on the type of problem you're trying to solve: classification or regression. Here's a breakdown:

Classification:

Goal: Predict discrete categories (e.g., image is a cat or dog, email is spam or not spam).
Output format: Probabilities of belonging to each class.
Suitable activation functions:
Softmax: Outputs a probability distribution over all possible classes, summing to 1. Ideal for multi-class problems (more than two).
Sigmoid: Outputs a probability between 0 and 1, suitable for binary classification (two classes).
Regression:

Goal: Predict continuous values (e.g., house price, temperature).
Output format: Direct numerical value representing the predicted value.
Suitable activation functions:
Linear: No transformation, outputs the weighted sum of inputs directly. Simple and interpretable, but might not capture complex relationships.
ReLU (Rectified Linear Unit): Outputs the input if it's positive, otherwise outputs 0. Fast and efficient, good for many regression tasks.
TanH: Similar to ReLU but outputs values between -1 and 1, can be useful in certain cases.
Additional factors to consider:

Complexity of the problem: More complex problems might benefit from non-linear activation functions like ReLU or tanh, even in regression tasks.
Computational efficiency: Linear and ReLU are generally faster than sigmoid or softmax.
Interpretability: Linear outputs are the easiest to interpret, while non-linear functions might be a "black box."
<img width="529" alt="image" src="https://github.com/Tijo2000/Machine-Learning/assets/56153083/a4e50d6d-1878-4e72-8dea-6a9c8dd5fd57">
<img width="375" alt="image" src="https://github.com/Tijo2000/Machine-Learning/assets/56153083/3c1a975b-0fbc-4760-853a-9347d4c77ac5">
<img width="515" alt="image" src="https://github.com/Tijo2000/Machine-Learning/assets/56153083/a948f253-f4f6-4e60-a546-ac1cecb947a5">

# Day 15
## why do we need Activation Functions
There are different activation function for different purpose some of the most commonly used are :
* Linear acitvation function(activation='linear')
This is used for regression problem where result is negative/positive.
* ReLU (activation = 'ReLU')
This is used for regression problem where result should be positive always and it is faster as compared to sigmoid function.
* Sigmoid function(activation='sigmoid')
It is used for classification problems where result must be on/off and it is slowere as compared to ReLU.
NOTE:For hidden layer we choose ReLU as activation and for output layer we choose activation according to our problems,because if we choose sigmoid in hidden layer than neural network becomes very slow so it better to choose Relu in hidden layer.
![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/713b73eb-ff6e-4b75-ae36-408457339a0e)


# Day 16
## Bias and Variance

![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/b48484fd-a076-4b6a-9d1b-fe3afb7d6cf4)

# Day 17
## Ethics of Machine Learning

*Precision - It tell of all positive prediction how many are actually positve.

*Recall - It tell of all real positive cases how many are actually predicted positive.

![image](https://github.com/Tijo2000/Machine-Learning/assets/56153083/e3362ae0-f6a8-4584-8397-327cfff47855)












