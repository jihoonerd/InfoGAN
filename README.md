# InfoGAN

Experiments with InfoGAN.


## Experiment 1) Split circle trajectory from whole data points

### Given data

![exp1](/assets/exp1/original.png)

### Generated data (InfoGAN)

* 2 discrete codes are used. [1, 0] for red and [0, 1] for green

![exp1_gen](/assets/exp1/infogan.gif)


## Experiment 2) Sequential prediction with given start(fixed), goal(fixed), and current position

* Starting point: Red
* Target point: Blue
* Current seen: Black
* Prediccted Next: Purple

Generator's input: `[start_x, start_y, target_x, target_y, cur_x, cur_y]`
Discriminator's input: `[(predicted/gt) x, (predicted/gt) y]`

Following single path is used for training and inference, but prediction is unstable.

### Original Path (equally distributed 30 points each for clock-wise and counter clock-wise direction)

![exp2](/assets/exp2/original.png)

### Generated data (Vanilla GAN)

![exp2_shuffle](/assets/exp2/w_shuffle.gif)

## Experiment 3) Sequential prediction with past context

* Starting point: Red
* Target point: Blue
* Current seen: Black
* Prediccted Next: Purple

Generator's input: `[start_x, start_y, target_x, target_y, cur_x, cur_y]`
Discriminator's input: `[start_x, start_y, target_x, target_y, previous_x, previous_y, (predicted/gt) x, (predicted/gt) y]`

Following single path is used for training and inference, but prediction is unstable.

### Original Path (equally distributed 30 points each for clock-wise and counter clock-wise direction)

![exp3](/assets/exp3/original.png)

### Generated data (Vanilla GAN)

![exp3_shuffle](/assets/exp3/w_shuffle.gif)





