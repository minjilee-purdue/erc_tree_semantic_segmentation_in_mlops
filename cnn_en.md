### CNN Overview
**Convolutional Neural Network (CNN)** is a representative deep learning model for learning image data. It extracts important features through convolution operations and learns progressively abstract information by applying pooling and activation functions.

The main processes of CNN are as follows:
1. Generate **feature maps** through **Convolution**
2. Apply **Activation Function (e.g., ReLU)**
3. Reduce dimensions through **Pooling (e.g., Max Pooling)**
4. Perform final prediction through **Fully Connected Layer (FC Layer)**

#### Convolution Operation
The core of CNN is the **convolution operation**, which is defined as the sum of element-wise multiplication between the input image $X$ and the filter (kernel) $W$.

$$
Y(i, j) = \sum_m \sum_n X(i+m, j+n) W(m, n)
$$

- $X(i, j)$: Input image pixel value
- $W(m, n)$: Convolution filter (kernel)
- $Y(i, j)$: Convolution result value (feature map)

In the case of RGB images, each pixel value has three channel values: Red (R), Green (G), and Blue (B). Therefore, $X(0,8)$ is not a single scalar value but a vector with 3 channel values.

**Pixel value representation in RGB images:** If $X(i,j)$ is an RGB image, the pixel value at a specific coordinate can be expressed as: $X(0,8)=(R,G,B)$

- $X(0,8,0)$ → Red
- $X(0,8,1)$ → Green
- $X(0,8,2)$ → Blue

RGB images are represented in the form of (Height, Width, Channels). For example, a 256 × 256 color image has the following 3D array (tensor) structure:

$$X \in \mathbb{R}^{256 \times 256 \times 3}$$

This means it can represent **3 channels (R, G, B) for a 256 x 256 pixel size**, and if $X(0,8)=(255,100,50)$, it is defined as **Red (R) value: 255, Green (G) value: 100, Blue (B) value: 50**.

#### Filter Movement Method in Convolution
Assuming the filter is placed at position $(i, j)$, let's set the filter size to 3×3. Select a 3×3 area around the position $(i, j)$. That is, take a 3×3 area that includes pixels above, below, left, and right of $(i, j)$.

#### Repeat Operation When Applying a 3×3 Size Filter
Multiply each element of the filter $W(m, n)$ by the pixel value $X(i + m, j + n)$ in the area covered by the filter in the input image. Add all these calculated values to generate the value $Y(i, j)$. The formula is as follows:

$$
Y(i, j) = \sum_{m=0}^{2} \sum_{n=0}^{2} X(i+m, j+n) W(m, n)
$$

In this formula, $i + m$ and $j + n$ represent the selected pixel positions within the filter. Since the filter size is 3×3, the range of $m$ and $n$ is 0~2 (i.e., 3 pixels).

Repeat $m = 0, 1, 2$ and $n = 0, 1, 2$ to multiply a total of 9 pixel values by the filter and add them together.

To summarize the CNN operation sequence again, it generally extracts features by repeating the following processes:

#### 1️⃣ Convolution Operation (Feature Map Generation)
* Generate feature maps by **applying filters (kernels)** to the input image.
* When the input size is fixed, the kernel size and output feature map size have an inverse relationship.
* The larger the feature map size (i.e., the smaller the kernel size), the more detailed information can be represented.
* During this process, important features such as **edges, textures, and patterns are learned**.
* Conclusion: In the convolutional layer, multiple filters (kernels) are used to extract features from the image. Each filter learns different features, and as many feature maps are generated as there are filters.
* These feature maps can contain both positive and negative values.

#### 2️⃣ Apply ReLU Activation Function
* Values after convolution operations **can be either negative or positive**.
* However, **negative values often do not represent features, so they are converted to 0**.
* "Activated feature maps" are generated through the activation function.
* The ReLU (Rectified Linear Unit) activation function works as follows:

$$f(x) = \max(0, x)$$

   * **Positive values are maintained**
   * **Negative values are converted to 0**

#### Why Apply ReLU?
   * **It adds non-linearity to help the model learn more complex patterns.**
   * **It removes negative values to prevent the vanishing gradient problem.**

#### 3️⃣ Pooling
* **A process that reduces the size of feature maps to decrease computation and preserve important features**.
* The most commonly used pooling method is **Max Pooling**:

$$Y(i, j) = \max_{m, n} X(i+m, j+n)$$

   * **Only keeps the maximum value** in a specific area (e.g., $2 \times 2$).
   * Ignores small changes and **maintains only the most important features**.
   * Helps the model **to be more generalized**, providing an **overfitting prevention effect**.

#### 4️⃣ Feature Map Generation Again → ReLU → Pooling...
* Apply multiple convolutional and pooling layers repeatedly to learn **increasingly complex features** from the image.
* For example:
   * First convolutional layer: **Edge detection** - In the initial layers of the network, large differences between adjacent pixel values are detected to extract image boundaries or contours. Example: Boundaries between bright and dark areas, areas with distinct color differences
   * Second convolutional layer: **Texture detection** - In deeper layers, edges come together to learn specific patterns or textures. Example: Fine repetitive structures of a lawn, texture of an object's surface
   * Third convolutional layer: **Learning parts of objects** - As the neural network deepens, it can recognize specific parts of objects, such as eyes, nose, and mouth, by combining texture information.
   * Last convolutional layer: **Whole object detection** - In the deepest layers, these parts combine to recognize entire objects (e.g., faces, cars, animals).

#### Overall Flow of CNN
Convolution → ReLU → Pooling → Convolution → ReLU → Pooling → ... → Fully Connected Layer → Softmax

1. Convolution (Feature Map Generation)
2. Apply ReLU Activation Function
3. Pooling (Dimension Reduction)
4. Convolution
5. Apply ReLU Activation Function
6. Pooling
7. ... Repeat ...
8. Fully Connected Network (FC Layer) → Softmax for Final Prediction

#### 5️⃣ Final Prediction Stage of CNN
The final class is predicted through flatten → fully connected layer (FC layer) → Softmax activation function.

The Softmax function returns probability values, and the formula is as follows:
$$P(y_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

$z_i$: Activation value of the neuron
$P(y_i)$: Probability for class $i$

At the end of the CNN model, there is usually a fully-connected layer (FC layer), with a flatten layer placed before it to convert the feature map into a 1D vector. Local features reflect information from specific areas, but for final classification, a probability distribution considering the overall context is needed. For this, CNN uses a fully-connected layer or global pooling to compress the feature map and calculate the final probabilities.

#### So Where Does Learning Occur?

Learning in CNN proceeds in two main stages: **Forward Propagation** and **Back Propagation**

In forward propagation:
- The filter (kernel) weights of the convolutional layer are learning parameters (initially, the filter weights are initialized with random values)
- When an input image is received, features are extracted with the current filter weights (for example, a 3x3 filter has 9 weight values, and these values are learning targets)
- The feature map obtained through the filter passes through ReLU and max pooling
- Finally, the classification result is output in the fully connected layer
- Calculate the difference (error) between this output value and the actual correct label

In back propagation:
- Calculate the gradient of the loss function based on the calculated error
- Update all weights in the network using this gradient (the aforementioned 9 weights are slightly modified in the back propagation process)
- The filter weights of the convolutional layer are gradually modified
- Through this process, filters evolve and learn to capture and detect important features (edges, textures, etc.)

That is, actual learning occurs in:
- Filter weights of the convolutional layer
- Weights of the fully connected layer
- Bias of each layer

These three elements are continuously updated through back propagation. Max pooling has no learning parameters, and ReLU simply applies a non-linear function, so no actual learning occurs. Looking at the flow of the learning process:

(1) Forward propagation: Input image → Convolution operation → ReLU → Max pooling. Repeat this process several times → Pass through fully connected layer → Output final prediction result

(2) Error calculation: Compare prediction value with actual label → Calculate loss function (e.g., cross-entropy)

(3) Back propagation: Calculate gradient for loss → Update all weights and biases using gradient descent

Through this process, the network gradually learns. By repeating this entire process for a specified number of epochs, the network develops to make increasingly accurate predictions.

---
#### Note: Since the filter (kernel) weights of the convolutional layer are learning parameters, there is a difference between convolution operations in mathematics and deep learning.

- In mathematics, convolution operation is defined for signal processing and uses flipped filters (kernels).
- In deep learning, convolution operation is used as a learnable operation to detect patterns (because feature extraction is important), so filters are not flipped.
- In deep learning, filters (kernels) are not fixed but are weights that are learned.
- That is, the operation used in CNN does not operate as a mathematical convolution, but in a way similar to **cross-correlation**.
