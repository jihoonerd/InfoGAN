# InfoGAN

Experiments with InfoGAN.


## Experiment 1) Split circle trajectory from whole data points

### Given data

![exp1](/assets/exp1/original.png)

### Generated data (InfoGAN)

* 2 discrete codes are used. [1, 0] for red and [0, 1] for green

![exp1_gen](/assets/exp1/infogan.gif)


## Experiment 2) Predict next position with given start(fixed), goal(fixed), and current position

* Starting point: Red
* Target point: Blue
* Current seen: Black
* Prediccted Next: Purple

Training unstable

### Experiment 2) Sequential Prediction (30 points each for clock-wise and counter clock-wise direction)

![exp2](/assets/exp2/original.png)

### Generated data (Vanilla GAN)

#### Without Shuffle

![exp2_wo_shuffle](/assets/exp2/wo_shuffle.gif)

#### With Shuffle

![exp2_shuffle](/assets/exp2/w_shuffle.gif)







