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



A multi-layered perceptron is able to learn an internal representation due
to the choice of a proper non-linear activation function in the end of each layer l.
This creates an input to the next layer of the network that cannot be reduced as a linear transformation of a particular feature transformation of the input x. This way the
network can “decide” its own transformation of the data in the lower layers in order to
simplify the separation of the dataset classes in upper layers. For this concrete case, the
single-layer perceptron is limited to the original feature representations (independent
pixel values), being unable to exploit correlations among pixels; the multi-layer perceptron can learn richer representations in its hidden units that correlate the information
from different pixels. If the activation function is linear, no additional expressiveness
is added and the multi-layer perceptron is as expressive as the single-layer one.


## b)
Final test accuracies:

Plot of train loss:

Plot of train and validation accuracies:




time python hw1-q1.py perceptron
Training epoch 1
train acc: 0.5118 | val acc: 0.5103
Training epoch 2
train acc: 0.4635 | val acc: 0.4573
Training epoch 3
train acc: 0.6308 | val acc: 0.6329
Training epoch 4
train acc: 0.3407 | val acc: 0.3408
Training epoch 5
train acc: 0.4342 | val acc: 0.4285
Training epoch 6
train acc: 0.5988 | val acc: 0.6018
Training epoch 7
train acc: 0.5679 | val acc: 0.5650
Training epoch 8
train acc: 0.6379 | val acc: 0.6353
Training epoch 9
train acc: 0.5988 | val acc: 0.5992
Training epoch 10
train acc: 0.5399 | val acc: 0.5367
Training epoch 11
train acc: 0.5582 | val acc: 0.5596
Training epoch 12
train acc: 0.4041 | val acc: 0.4035
Training epoch 13
train acc: 0.5813 | val acc: 0.5804
Training epoch 14
train acc: 0.6054 | val acc: 0.6074
Training epoch 15
train acc: 0.5916 | val acc: 0.5884
Training epoch 16
train acc: 0.5556 | val acc: 0.5509
Training epoch 17
train acc: 0.5752 | val acc: 0.5798
Training epoch 18
train acc: 0.5937 | val acc: 0.5961
Training epoch 19
train acc: 0.4873 | val acc: 0.4831
Training epoch 20
train acc: 0.4654 | val acc: 0.4610
Final test acc: 0.3422

real    2m35.029s
user    0m0.016s
sys     0m0.000s


time python hw1-q1.py logistic_regression -epochs 50 -learning_rate 0.01
Training epoch 1
train acc: 0.6103 | val acc: 0.6052
Training epoch 2
train acc: 0.5548 | val acc: 0.5563
Training epoch 3
train acc: 0.6490 | val acc: 0.6493
Training epoch 4
train acc: 0.6459 | val acc: 0.6466
Training epoch 5
train acc: 0.5968 | val acc: 0.5950
Training epoch 6
train acc: 0.6414 | val acc: 0.6445
Training epoch 7
train acc: 0.6528 | val acc: 0.6544
Training epoch 8
train acc: 0.6556 | val acc: 0.6563
Training epoch 9
train acc: 0.6468 | val acc: 0.6495
Training epoch 10
train acc: 0.6335 | val acc: 0.6300
Training epoch 11
train acc: 0.6419 | val acc: 0.6386
Training epoch 12
train acc: 0.6028 | val acc: 0.5980
Training epoch 13
train acc: 0.6490 | val acc: 0.6535
Training epoch 14
train acc: 0.6511 | val acc: 0.6526
Training epoch 15
train acc: 0.6622 | val acc: 0.6604
Training epoch 16
train acc: 0.6452 | val acc: 0.6464
Training epoch 17
train acc: 0.6382 | val acc: 0.6389
Training epoch 18
train acc: 0.6020 | val acc: 0.6054
Training epoch 19
train acc: 0.6442 | val acc: 0.6459
Training epoch 20
train acc: 0.6205 | val acc: 0.6229
Training epoch 21
train acc: 0.6413 | val acc: 0.6438
Training epoch 22
train acc: 0.6478 | val acc: 0.6437
Training epoch 23
train acc: 0.6571 | val acc: 0.6562
Training epoch 24
train acc: 0.6438 | val acc: 0.6408
Training epoch 25
train acc: 0.6594 | val acc: 0.6583
Training epoch 26
train acc: 0.6575 | val acc: 0.6578
Training epoch 27
train acc: 0.6577 | val acc: 0.6604
Training epoch 28
train acc: 0.6529 | val acc: 0.6569
Training epoch 29
train acc: 0.6502 | val acc: 0.6557
Training epoch 30
train acc: 0.6622 | val acc: 0.6555
Training epoch 31
train acc: 0.6173 | val acc: 0.6197
Training epoch 32
train acc: 0.6618 | val acc: 0.6568
Training epoch 33
train acc: 0.6511 | val acc: 0.6457
Training epoch 34
train acc: 0.6600 | val acc: 0.6613
Training epoch 35
train acc: 0.6559 | val acc: 0.6522
Training epoch 36
train acc: 0.6646 | val acc: 0.6669
Training epoch 37
train acc: 0.6637 | val acc: 0.6619
Training epoch 38
train acc: 0.6356 | val acc: 0.6373
Training epoch 39
train acc: 0.5191 | val acc: 0.5153
Training epoch 40
train acc: 0.6145 | val acc: 0.6099
Training epoch 41
train acc: 0.6472 | val acc: 0.6523
Training epoch 42
train acc: 0.6423 | val acc: 0.6376
Training epoch 43
train acc: 0.6509 | val acc: 0.6481
Training epoch 44
train acc: 0.6679 | val acc: 0.6641
Training epoch 45
train acc: 0.6497 | val acc: 0.6491
Training epoch 46
train acc: 0.5570 | val acc: 0.5564
Training epoch 47
train acc: 0.6655 | val acc: 0.6660
Training epoch 48
train acc: 0.5834 | val acc: 0.5787
Training epoch 49
train acc: 0.5806 | val acc: 0.5781
Training epoch 50
train acc: 0.6609 | val acc: 0.6568
Final test acc: 0.5784

real    3m46.399s
user    0m0.000s
sys     0m0.015s

time python hw1-q1.py logistic_regression -epochs 50 -learning_rate 0.001
Training epoch 1
train acc: 0.6094 | val acc: 0.6067
Training epoch 2
train acc: 0.6312 | val acc: 0.6339
Training epoch 3
train acc: 0.6352 | val acc: 0.6369
Training epoch 4
train acc: 0.6407 | val acc: 0.6444
Training epoch 5
train acc: 0.6291 | val acc: 0.6262
Training epoch 6
train acc: 0.6444 | val acc: 0.6459
Training epoch 7
train acc: 0.6460 | val acc: 0.6464
Training epoch 8
train acc: 0.6493 | val acc: 0.6520
Training epoch 9
train acc: 0.6468 | val acc: 0.6484
Training epoch 10
train acc: 0.6516 | val acc: 0.6506
Training epoch 11
train acc: 0.6507 | val acc: 0.6514
Training epoch 12
train acc: 0.6539 | val acc: 0.6546
Training epoch 13
train acc: 0.6524 | val acc: 0.6496
Training epoch 14
train acc: 0.6553 | val acc: 0.6555
Training epoch 15
train acc: 0.6576 | val acc: 0.6581
Training epoch 16
train acc: 0.6555 | val acc: 0.6556
Training epoch 17
train acc: 0.6490 | val acc: 0.6483
Training epoch 18
train acc: 0.6512 | val acc: 0.6526
Training epoch 19
train acc: 0.6557 | val acc: 0.6574
Training epoch 20
train acc: 0.6564 | val acc: 0.6575
Training epoch 21
train acc: 0.6546 | val acc: 0.6550
Training epoch 22
train acc: 0.6568 | val acc: 0.6559
Training epoch 23
train acc: 0.6583 | val acc: 0.6592
Training epoch 24
train acc: 0.6566 | val acc: 0.6568
Training epoch 25
train acc: 0.6605 | val acc: 0.6619
Training epoch 26
train acc: 0.6606 | val acc: 0.6615
Training epoch 27
train acc: 0.6570 | val acc: 0.6577
Training epoch 28
train acc: 0.6576 | val acc: 0.6597
Training epoch 29
train acc: 0.6572 | val acc: 0.6599
Training epoch 30
train acc: 0.6609 | val acc: 0.6629
Training epoch 31
train acc: 0.6577 | val acc: 0.6608
Training epoch 32
train acc: 0.6586 | val acc: 0.6588
Training epoch 33
train acc: 0.6580 | val acc: 0.6577
Training epoch 34
train acc: 0.6601 | val acc: 0.6619
Training epoch 35
train acc: 0.6603 | val acc: 0.6581
Training epoch 36
train acc: 0.6610 | val acc: 0.6609
Training epoch 37
train acc: 0.6607 | val acc: 0.6605
Training epoch 38
train acc: 0.6550 | val acc: 0.6569
Training epoch 39
train acc: 0.6497 | val acc: 0.6538
Training epoch 40
train acc: 0.6625 | val acc: 0.6648
Training epoch 41
train acc: 0.6588 | val acc: 0.6602
Training epoch 42
train acc: 0.6604 | val acc: 0.6630
Training epoch 43
train acc: 0.6629 | val acc: 0.6640
Training epoch 44
train acc: 0.6601 | val acc: 0.6602
Training epoch 45
train acc: 0.6625 | val acc: 0.6661
Training epoch 46
train acc: 0.6525 | val acc: 0.6560
Training epoch 47
train acc: 0.6626 | val acc: 0.6634
Training epoch 48
train acc: 0.6566 | val acc: 0.6562
Training epoch 49
train acc: 0.6623 | val acc: 0.6627
Training epoch 50
train acc: 0.6625 | val acc: 0.6639
Final test acc: 0.5936

real    4m5.099s
user    0m0.016s
sys     0m0.000s


time python hw1-q1.py mlp
Training epoch 1
loss: 7.6605 | train acc: 0.5317 | val acc: 0.5344
Training epoch 2
loss: 1.1810 | train acc: 0.6590 | val acc: 0.6564
Training epoch 3
loss: 0.9844 | train acc: 0.6738 | val acc: 0.6712
Training epoch 4
loss: 0.9115 | train acc: 0.6866 | val acc: 0.6820
Training epoch 5
loss: 0.8579 | train acc: 0.7022 | val acc: 0.7001
Training epoch 6
loss: 0.8190 | train acc: 0.7194 | val acc: 0.7145
Training epoch 7
loss: 0.7839 | train acc: 0.7364 | val acc: 0.7334
Training epoch 8
loss: 0.7584 | train acc: 0.7371 | val acc: 0.7299
Training epoch 9
loss: 0.7341 | train acc: 0.7439 | val acc: 0.7409
Training epoch 10
loss: 0.7126 | train acc: 0.7599 | val acc: 0.7531
Training epoch 11
loss: 0.6959 | train acc: 0.7628 | val acc: 0.7566
Training epoch 12
loss: 0.6781 | train acc: 0.7622 | val acc: 0.7553
Training epoch 13
loss: 0.6646 | train acc: 0.7732 | val acc: 0.7624
Training epoch 14
loss: 0.6518 | train acc: 0.7684 | val acc: 0.7618
Training epoch 15
loss: 0.6398 | train acc: 0.7825 | val acc: 0.7710
Training epoch 16
loss: 0.6296 | train acc: 0.7757 | val acc: 0.7663
Training epoch 17
loss: 0.6188 | train acc: 0.7842 | val acc: 0.7725
Training epoch 18
loss: 0.6097 | train acc: 0.7898 | val acc: 0.7761
Training epoch 19
loss: 0.6004 | train acc: 0.7915 | val acc: 0.7806
Training epoch 20
loss: 0.5939 | train acc: 0.7988 | val acc: 0.7845
Final test acc: 0.7486