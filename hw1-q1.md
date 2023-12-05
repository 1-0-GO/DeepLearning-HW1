# 1 
## a) 
Accuracies:
- Training:
- Validation:
- Test:

Plot:

## b)
Test accuracy:

Plot of validation and train accuracies:
- For learning rate = 0.01


- For learning rate = 0.001

Comparison between the two models based on the previous plots:


# 2
## a)
There are two things mentioned here that will discussed - expressiveness and training complexity.

Expressiveness:

Logistic Regression: This model is a linear classifier. It is using pixel values as features, and attempts to separate classes using a linear decision boundary. Its expressiveness is limited to representing linear relationships in the data. It cannot model more complex patterns between the pixel values (which are the input features to the model).

Multi-Layer Perceptron with ReLU Activations: MLPs are capable of modeling non-linear relationships. The ReLU activation function introduces non-linearity into the network which, combined with multiple layers, allows the MLP to learn more complex patterns and interactions between the pixel values (which are the input features to the model) than logistic regression can. Essentially, it constructs much more rich features before the final linear classification layer. This final layer if it uses softmax activation and logistic loss behaves the same way as logistic regression, only now it receives these more expressive learned features. 

Training Complexity:

Logistic Regression: The optimization problem in logistic regression is convex, meaning there is a single global minimum. Gradient descent methods are guaranteed to converge to this global minimum. This makes training logistic regression models relatively straightforward and computationally less intensive.

Multi-Layer Perceptron: Training MLPs is more complex. The presence of multiple layers and non-linear activations turns the optimization problem into a non-convex one. The optimization landscape is a lot more complicated. This means there can be multiple local minima, and gradient descent methods might not necessarily converge to the global minimum. Training MLPs requires careful tuning of parameters (like learning rates, initialization, etc.) and is more computationally intensive.

Let's recap: while MLPs with ReLU activations are more expressive and capable of capturing complex patterns in data like images, their training process is more complex and computationally demanding compared to logistic regression, which is easier to train due to its convex optimization landscape but is less expressive due to its linear nature.

## b)
Final test accuracies:

Plot of train loss:

Plot of train and validation accuracies: