The above code shows my implementation of BNNs using Metropolis-Hastings algorithm.
The entire implementation uses numpy for its operations.
For the following I have made a 3 layer network with weights of size [2,3],[3,3],[1,3] respectively.
Each layer is connected with the sigmoid function.
All the weights are initialised using a standard normal distibution. This was chosen arbritrarily but it should not affect the performance too much.
For the Metropolis-Hastings algorithm, a new set of weights is draw for each step in which the mean is the same as the previous weights with standard deviation 1. 
The likelihood has been taken as the probability of getting the y_pred if y is the meand and standard deviation 1.
The likelihood has been taken arbitrarily as it is the standard and can be investigated upon to improve performance.

One peculiar finding about the metropolis hastings is its similarity to the normal method of optimisation.
If taken the log of likelihoods and priors we can see that they are analogous to that of MSE loss with l2 regularization.
This the algorithm can also be understood as taking a random step based on teh previous model, calculating the loss and choosing the lower of the two.
Considering about the form we can tell how this algorithm is worse as compared to gradient descent and the reason for its lack of popularity.
Since there arent any gradients the model takes steps (transition) 'blindly' without the surity of improving performance.

Another point to note would be the use of prior.
The prior can also be used to instill pre-defined knowledge to the model if we have any. 
This was experimented in the code by training the model taking the mean of prior to be 0 and then training the model again with the mean of the prior being equal to that of the parameters of the old model.
I was able to achieve an accuracy score of 93 percent in 20000 epochs on the test set.
This is substantially higher than the 83-84 percent accuracy we achieve without changing the prior.
Despite the increased performance the implementation was scrapped as I was unable to recreate the results consistantly and was unable to explore further due to lack of time.

Another observation was the increased consistancy in the results after adding the prior to the likelidood than without it. Though I am not sure of the reason for the same, I believe it can be attributed to bounding the weights to stay close to a particular value (in our case 0) forcing the network to give similar results.

For improvement fursther improvement and exploration,
1) We could make the standard deviation dependent on the weights and data as well which is currently set to 1 everywhere.
2) We can look into the correct initialization of prior for a way to give pre-known knowledge to our model.
3) We can explore other likelihood parameters which may better define the problem at hand.