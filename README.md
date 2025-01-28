# multilayer-perceptron
http://dossierslmm.chez-alice.fr/fiche/tableaux_derivees.pdf


## perceptron (simple neuron)

### functions

#### linear model function
$z(x_1, x_2)=x_1*w_1 + x_2*w_2 + b$  
Where :
- $x_1$ and $x_2$ are data passed to the fonction (from the data).  
- $w_1$ and $w_2$ are weights determined during the learning phase to make a parameter more important than an another
- $b$ the bias is also determined during the leanrning phase to make the program able to shift to the activation fonction [here](https://stackoverflow.com/questions/2480650/what-is-the-role-of-the-bias-in-neural-networks)  

The result of this function tell us on which side of the data we are (logistic regression).  
If it is positive we are on the right side of the fonction, if it is negative on the left side.  

FAIRE UN DESSIN ICI AVEC DROITE QUI SEPARE DES POINTS

#### activation function
The activation function is used to convert the result of $z$ to a probability.  
There are multiple activations functions but in this case we use the [**sigmoide**](https://en.wikipedia.org/wiki/Sigmoid_function).  

$a(z)=\frac{1}{1+e^-z}$  

FAIRE UN DESSIN AVEC LA FONCTION SIGMOIDE

#### log loss function
The function to calculate the maximum of likelihood is represented by $L=\prod_{i=0}^m a_{i}^{a_y}*(1-a_i)^{1-y_i}$, it is the sum of your probabilities.  
The issue is that the number will tend towards 0 due to multiplication of number between [0, 1].  
To avoid this problem we will use the $\log$ and use [logarithms](https://en.wikipedia.org/wiki/Logarithm) attributes.  
At this moment, this function have a maximum that represent the *best of the likelihood*.  
If we expand and simplify this function it will be $LL$

$LL=-\frac{1} {m}\sum_{i=0}^m y_i\log(a_i)+(1-y_i)\log(1-a_i)$

You might wonder what is the $-\frac{1}{m}$ it is because we can't maximise a function in maths. So instead of trying to maximise $L$ we are going to minimise $-L$, which is the same thing.  
The division by m is just there to do some normalization of our result.  

> [!IMPORTANT]
> This function $LL$ is called LogLoss

> [!NOTE]
> Bernoulli distribution

If we want to minimize our Log Loss function we can use the **gradient descent** algorithm.  
This involves to calculate the derivative of Log Loss.

> [!TIP]
> You should do a linear regression before starting this project

### derivated functions
Few links to understand some tips about derivative and maths :
- [Chain Rule](https://en.wikipedia.org/wiki/Chain_rule)
- [Derivative](https://en.wikipedia.org/wiki/Derivative)
- [List usuals derivatives](http://dossierslmm.chez-alice.fr/fiche/tableaux_derivees.pdf)]

