from __future__ import print_function
import numpy as np


class RBM:

    def __init__(self, tup, num_visible=1034):
        np_rng = np.random.RandomState(1234)
        self.num_visible = num_visible
        self.debug_print = True

        self.l = len(tup)
        print("There are",self.l,"hidden layers with",tup,"neurons each")
        num_hidden1 = tup[0]
        self.num_hidden1 = num_hidden1
        # weights of the 1st hidden layer
        self.weights1 = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden1 + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden1 + num_visible)),
            size=(num_visible, num_hidden1)))
        self.weights1 = np.insert(self.weights1, 0, 0, axis=0)
        self.weights1 = np.insert(self.weights1, 0, 0, axis=1)

        if self.l > 1:
            num_hidden2 = tup[1]
            self.num_hidden2 = num_hidden2
            # weights of the 2nd hidden layer
            self.weights2 = np.asarray(np_rng.uniform(
                low=-0.1 * np.sqrt(6. / (num_hidden1 + num_hidden2)),
                high=0.1 * np.sqrt(6. / (num_hidden1 + num_hidden2)),
                size=(num_hidden1, num_hidden2)))
            self.weights2 = np.insert(self.weights2, 0, 0,
                                      axis=0)  # Insert weights for the bias units into the first row and first column.
            self.weights2 = np.insert(self.weights2, 0, 0, axis=1)
        if self.l > 2:
            num_hidden3 = tup[2]
            self.num_hidden3 = num_hidden3
            # weights of the 2nd hidden layer
            self.weights3 = np.asarray(np_rng.uniform(
                low=-0.1 * np.sqrt(6. / (num_hidden2 + num_hidden3)),
                high=0.1 * np.sqrt(6. / (num_hidden2 + num_hidden3)),
                size=(num_hidden2, num_hidden3)))
            self.weights3 = np.insert(self.weights3, 0, 0, axis=0)
            self.weights3 = np.insert(self.weights3, 0, 0, axis=1)
        if self.l > 3:
            num_hidden4 = tup[3]
            self.num_hidden4 = num_hidden4
            # weights of the 2nd hidden layer
            self.weights4 = np.asarray(np_rng.uniform(
                low=-0.1 * np.sqrt(6. / (num_hidden3 + num_hidden4)),
                high=0.1 * np.sqrt(6. / (num_hidden3 + num_hidden4)),
                size=(num_hidden3, num_hidden4)))
            self.weights4 = np.insert(self.weights4, 0, 0, axis=0)
            self.weights4 = np.insert(self.weights4, 0, 0, axis=1)

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
        # and sqrt(6. / (num_hidden + num_visible)). One could vary the
        # standard deviation by multiplying the interval with appropriate value.
        # Here we initialize the weights with mean 0 and standard deviation 0.1.
        # Reference: Understanding the difficulty of training deep feedforward
        # neural networks by Xavier Glorot and Yoshua Bengio
        '''
        print(self.weights1.shape, "3333")
        print(self.weights2.shape, "3333")
        print(self.weights3.shape, "3333")
        print(self.weights4.shape, "3333")
        '''


    def train(self, data, max_epochs=3, learning_rate=0.1):
        """
    Train the machine.

    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """

        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis=1)

        for epoch in range(max_epochs):
            # Clamp to the data and sample from the hidden units.
            # (This is the "positive CD phase", aka the reality phase.)
            # stage: 1st layer forward
            pos_hidden1_activations = np.dot(data, self.weights1)
            pos_hidden1_probs = self._logistic(pos_hidden1_activations)
            pos_hidden1_probs[:, 0] = 1  # Fix the bias unit.
            pos_hidden1_states = pos_hidden1_probs > np.random.rand(num_examples, self.num_hidden1 + 1)
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
            # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
            # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            pos_associations = np.dot(data.T, pos_hidden1_probs)

            if self.l > 1:
                # stage: 2nd layer forward
                num_examples2 = pos_hidden1_states.shape[0]
                pos_hidden2_activations = np.dot(pos_hidden1_states, self.weights2)
                pos_hidden2_probs = self._logistic(pos_hidden2_activations)
                pos_hidden2_probs[:, 0] = 1
                pos_hidden2_states = pos_hidden2_probs > np.random.rand(num_examples2, self.num_hidden2 + 1)

                pos_associations2 = np.dot(pos_hidden1_states.T, pos_hidden2_probs)
            if self.l > 2:
                # stage: 3rd layer forward
                num_examples3 = pos_hidden2_states.shape[0]
                pos_hidden3_activations = np.dot(pos_hidden2_states, self.weights3)
                pos_hidden3_probs = self._logistic(pos_hidden3_activations)
                pos_hidden3_probs[:, 0] = 1
                pos_hidden3_states = pos_hidden3_probs > np.random.rand(num_examples3, self.num_hidden3 + 1)

                pos_associations3 = np.dot(pos_hidden2_states.T, pos_hidden3_probs)

            if self.l > 3:
                # stage: 4th layer forward
                num_examples4 = pos_hidden3_states.shape[0]
                pos_hidden4_activations = np.dot(pos_hidden3_states, self.weights4)
                pos_hidden4_probs = self._logistic(pos_hidden4_activations)
                pos_hidden4_probs[:, 0] = 1
                pos_hidden4_states = pos_hidden4_probs > np.random.rand(num_examples4, self.num_hidden4 + 1)

                pos_associations4 = np.dot(pos_hidden3_states.T, pos_hidden4_probs)

                # stage: 3rd layer Backwards
                neg_hidden3_activations = np.dot(pos_hidden4_states, self.weights4.T)
                neg_hidden3_probs = self._logistic(neg_hidden3_activations)
                neg_hidden3_probs[:, 0] = 1  # Fix the bias unit.
                neg_hidden4_activations = np.dot(neg_hidden3_probs, self.weights4)
                neg_hidden4_probs = self._logistic(neg_hidden4_activations)

                neg_associations4 = np.dot(neg_hidden3_probs.T, neg_hidden4_probs)

                self.weights4 += learning_rate * ((pos_associations4 - neg_associations4) / num_examples4)

            if self.l > 2:
                # stage: 3rd layer Backwards
                neg_hidden2_activations = np.dot(pos_hidden3_states, self.weights3.T)
                neg_hidden2_probs = self._logistic(neg_hidden2_activations)
                neg_hidden2_probs[:, 0] = 1  # Fix the bias unit.
                neg_hidden3_activations = np.dot(neg_hidden2_probs, self.weights3)
                neg_hidden3_probs = self._logistic(neg_hidden3_activations)

                neg_associations3 = np.dot(neg_hidden2_probs.T, neg_hidden3_probs)

                self.weights3 += learning_rate * ((pos_associations3 - neg_associations3) / num_examples3)

            if self.l > 1:
                # reconstruct 1st hidden layer
                # stage: 2nd layer Backwards
                neg_hidden1_activations = np.dot(pos_hidden2_states, self.weights2.T)
                neg_hidden1_probs = self._logistic(neg_hidden1_activations)
                neg_hidden1_probs[:, 0] = 1  # Fix the bias unit.
                neg_hidden2_activations = np.dot(neg_hidden1_probs, self.weights2)
                neg_hidden2_probs = self._logistic(neg_hidden2_activations)

                neg_associations2 = np.dot(neg_hidden1_probs.T, neg_hidden2_probs)
                self.weights2 += learning_rate * ((pos_associations2 - neg_associations2) / num_examples2)

            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            # stage: 1st layer Backwards
            neg_visible_activations = np.dot(pos_hidden1_states, self.weights1.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:, 0] = 1  # Fix the bias unit.
            neg_hidden1_activations = np.dot(neg_visible_probs, self.weights1)
            neg_hidden1_probs = self._logistic(neg_hidden1_activations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states
            # themselves.
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden1_probs)

            # Update weights.
            self.weights1 += learning_rate * ((pos_associations - neg_associations) / num_examples)

            error = np.sum((data - neg_visible_probs) ** 2)
            if self.debug_print:
                print("Epoch %s: error is %s" % (epoch, error))

    def run_visible(self, data):
        """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.
    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """

        # stage: 1st layer
        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the hidden units (plus a bias unit)
        # sampled from a training example.
        hidden_states = np.ones((num_examples, self.num_hidden1 + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(data, self.weights1)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._logistic(hidden_activations)
        # Turn the hidden units on with their specified probabilities.
        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.num_hidden1 + 1)
        # Always fix the bias unit to 1.
        hidden_states[:, 0] = 1
        if self.l==1:
            # Ignore the bias units.
            hidden_states = hidden_states[:, 1:]
            return hidden_states
        # stage: 2nd layer
        num_examples2 = hidden_states.shape[0]
        hidden2_states = np.ones((num_examples2, self.num_hidden2 + 1))
        hidden2_activations = np.dot(hidden_states, self.weights2)
        hidden2_probs = self._logistic(hidden2_activations)
        hidden2_states[:, :] = hidden2_probs > np.random.rand(num_examples2, self.num_hidden2 + 1)
        hidden2_states[:, 0] = 1
        if self.l==2:
            hidden2_states = hidden2_states[:, 1:]
            return hidden2_states

        # stage: 3rd layer
        num_examples3 = hidden2_states.shape[0]
        hidden3_states = np.ones((num_examples3, self.num_hidden3 + 1))
        hidden3_activations = np.dot(hidden2_states, self.weights3)
        hidden3_probs = self._logistic(hidden3_activations)
        hidden3_states[:, :] = hidden3_probs > np.random.rand(num_examples3, self.num_hidden3 + 1)
        hidden3_states[:, 0] = 1
        if self.l==3:
            hidden3_states = hidden3_states[:, 1:]
            return hidden3_states

        num_examples4 = hidden3_states.shape[0]
        hidden4_states = np.ones((num_examples4, self.num_hidden4 + 1))
        hidden4_activations = np.dot(hidden3_states, self.weights4)
        hidden4_probs = self._logistic(hidden4_activations)
        hidden4_states[:, :] = hidden4_probs > np.random.rand(num_examples4, self.num_hidden4 + 1)
        hidden4_states[:, 0] = 1
        if self.l==4:
            hidden4_states = hidden4_states[:, 1:]
            return hidden4_states

    # TODO: Remove the code duplication between this method and `run_visible`?
    def run_hidden(self, data):
        """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.

    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the visible units (plus a bias unit)
        # sampled from a training example.
        visible_states = np.ones((num_examples, self.num_visible + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations of the visible units.
        visible_activations = np.dot(data, self.weights1.T)
        # Calculate the probabilities of turning the visible units on.
        visible_probs = self._logistic(visible_activations)
        # Turn the visible units on with their specified probabilities.
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
        # Always fix the bias unit to 1.
        # visible_states[:,0] = 1

        # Ignore the bias units.
        visible_states = visible_states[:, 1:]
        return visible_states

    def daydream(self, num_samples):
        """
    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.

    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

        # Create a matrix, where each row is to be a sample of of the visible units
        # (with an extra bias unit), initialized to all ones.
        samples = np.ones((num_samples, self.num_visible + 1))

        # Take the first sample from a uniform distribution.
        samples[0, 1:] = np.random.rand(self.num_visible)

        # Start the alternating Gibbs sampling.
        # Note that we keep the hidden units binary states, but leave the
        # visible units as real probabilities. See section 3 of Hinton's
        # "A Practical Guide to Training Restricted Boltzmann Machines"
        # for more on why.
        for i in range(1, num_samples):
            visible = samples[i - 1, :]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights1)
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden1 + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights1.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i, :] = visible_states

        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:, 1:]

    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    r = RBM(num_visible=6, num_hidden1=2)
    training_data = np.array(
        [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0]])
    r.train(training_data, max_epochs=5000)
    print(r.weights1)
    user = np.array([[0, 0, 0, 1, 1, 0]])
    print(r.run_visible(user))
