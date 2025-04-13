#Data Notes
<h1> Goal: to learn TensorFlow and predict the strength of concrete with Neural Networks</h1>

<h2>Things to do</h2>

<hr>

<h2>Helpful Libraries</h2>
<ul>
    <li>TensorFlow</li>
    <li>TensorFlow-metal</li>
    <li>TensorFlow-macos</li>
    <li>Keras</li>
    <li>matplotlib</li>
</ul>

<h2> Deep Learning Section for Keras </h2>
<ul>
    <li>You could think of each layer in a neural network as performing some kind of relatively simple transformation.</li>
    <li>Without activation functions, neural networks can only learn linear relationships. In order to fit curves, we'll need to use activation functions.</li>
    <li>The layers before the output layer are sometimes called hidden since we never see their outputs directly.</li>
    <li>The Sequential model we've been using will connect together a list of layers in order from first to last: the first layer gets the input, the last layer produces the output.</li>
</ul>

<h3>Types of Activation layers</h3>
<ul>
    <li>relu</li>
    <li>softmax</li>
    <li>sigmoid</li>
    <li>selu</li>
</ul>
<h3>Example: layers</h3>

<pre>
model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
</pre>

<h2>Note 1: input shape</h2>
<p>if you code the input shape as shown above you will get a warning telling you to a better way to do it. <url>https://keras.io/guides/sequential_model/</url> says you can build your model then call model.add Input(shape) and this can start your model with a shape. In general, it's a recommended best practice to always specify the input shape of a Sequential model in advance if you know what it is.</p>

<hr>

<h3>Sequential Models</h3>
<p>A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. reference keras.io</p>
dont use a sequential model if your layers have multiple inputs or expect mutiple outputs

<hr>

<p>up until now we have just set the foundation for a network. We still need to tell the network what problem to solve that is where we use the loss function</p>

<strong>The loss function measures the difference between the the target's true value and the models predicted value.</strong>

<h3>Types of loss function for Regression</h3>
<ul>
    <li>Mean Absolute Error</li>
    <li>Mean Squared Error</li>
    <li>Huber</li>
</ul>

<p>During Training the loss function will guide the model</p>

<h3>The Optimizer</h3>

<p>The Optimizer is an algorithm that adjusts the weights to minimize loss. Most algorithms used in deep learning are known as <strong>stochastic gradient descent</strong>. They are iterative.</p>

<p>Adding the Loss and Optimizer comes after defining a model, you can add a loss function and optimizer with the model's compile method</p>

<pre>
model.compile(
    optimizer="adam",
    loss="mae",
)
</pre>

<pre>
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)
</pre>

<h2>Note: EPOCH</h2>
<p>An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model. Another way to define an epoch is the number of passes a training dataset takes around an algorithm.</p>

<h2>Note: Plotting Loss</h2>
<p>After training your model you can view the loss through the 'history.history' Convert to df and plot</p>

<h3>Capacity & Stopping</h3>

<p>You can increase the capacity of a network either by making it wider (more units to existing layers) or by making it deeper (adding more layers). <br>
Wider networks have an easier time learning more linear relationships, while deeper networks prefer more nonlinear ones.</p>

<pre>
model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])

wider = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])

deeper = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])
</pre>