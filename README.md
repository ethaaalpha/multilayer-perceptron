# multilayer-perceptron
http://dossierslmm.chez-alice.fr/fiche/tableaux_derivees.pdf


### perceptron (simple neuron)

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


$L=-\frac{1} {m}\sum_{i=0}^m y_i\log(a_i)+(1-y_i)\log(1-a_i)$