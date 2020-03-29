import tensorflow as tf
import numpy as np
import PIL.Image
import time

content_path = './results/man_source.jpg'
style_path = './results/spaghetti_source.jpg'


# Converts 4 dimensional 0-1 float32 RGB matrix to Pillow image
def tensor_to_image(tensor):
    tensor = tensor * 255
    # Convert float32 to uint8 matrix
    tensor = np.array(tensor, dtype=np.uint8)
    # Take first image matrix
    tensor = tensor[0]
    # Convert it to Pillow image
    return PIL.Image.fromarray(tensor)


# Loads image file, scales it to max 512px at the longest dimension
# Returns scaled image as 3-dimensional tensor - number of channels X width X height
def load_img(path_to_img):
    # Maximum length in pixels along the longest dimension
    max_dim = 512
    # Read raw file as byte array
    img = tf.io.read_file(path_to_img)
    # Convert byte array to 3-dimensional matrix of RGB pixels from 0 to 255
    img = tf.image.decode_image(img, channels=3)
    # Convert 3-dimensional matrix of RGB pixels from 0 to 255 to 0 to 1
    # because it is what our neural network expects
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Determine shape of our image to be able to rescale it
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    # Get length along the longest dimension
    long_dim = max(shape)
    # Determine scale along that dimension
    scale = max_dim / long_dim
    # Determine new scaled shape of our image
    new_shape = tf.cast(shape * scale, tf.int32)
    # Resize image to new scaled size
    img = tf.image.resize(img, new_shape)
    # Convert to 4-dimensional matrix (first dimensional is 1) - number of images in batch
    img = img[tf.newaxis, :]
    return img


content_image = load_img(content_path)
style_image = load_img(style_path)

# Those layers will be used to determine general
# details in our content image - contours, figures, etc
content_layers = ['block5_conv2',
                  'block5_conv3',
                  'block5_conv4']

# Those layers will be used to determine the overall "style" of our style image
# Output of each convolutional layer will be passed on to compute gram matrix which tells us how evenly
# each feature spreads across style image - if it was captured by other filters on that layer
# If that feature spreads evenly across style image - it will be transferred to our content image
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def gram_matrix(input_tensor):
    input_shape = tf.shape(input_tensor)
    # Get the number of feature channels for the input tensor,
    # which is assumed to be from a convolutional layer with 4-dim.
    num_channels = int(input_shape[3])
    # Reshape the tensor so it is a 2-dim matrix. This essentially
    # flattens the contents of each feature-channel.
    matrix = tf.reshape(input_tensor, shape=[-1, num_channels])
    # Calculate the Gram-matrix as the matrix-product of
    # the 2-dim matrix with itself. This calculates the
    # dot-products of all combinations of the feature-channels.
    gram = tf.matmul(tf.transpose(matrix), matrix)
    # Average that outer product over all locations
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return gram / num_locations


# Load pretrained VGG19 model without fully-connected layer at the top of the network
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
# Make network not trainable
vgg.trainable = False
# All style and content layers will be output of our model
outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
# Construct model
model = tf.keras.Model([vgg.input], outputs)
# Make it non trainable
model.trainable = False


# Processing one image (3-d tensor in 0-1 float32 RGB) by our model
def style_transfer(inputs):
    # Converts RGB 0-1 float32 tensor to 0-255
    inputs = inputs * 255.0
    # Converts our RGB 0 to 255 tensor to BGR -1 to 1 tensor suitable for ImageNet network
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    # Process preprocessed image by network
    outputs = model(preprocessed_input)
    # Take output of each layer
    style_outputs, content_outputs = (outputs[:num_style_layers], outputs[num_style_layers:])
    # For each output from style layer - compute gram matrix
    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
    # Map each output tensor to corresponding content layer
    content_dict = {content_name: value for content_name, value in zip(content_layers, content_outputs)}
    # Map each output tensor to corresponding style layer
    style_dict = {style_name: value for style_name, value in zip(style_layers, style_outputs)}
    return {'content': content_dict, 'style': style_dict}


# Process style image and get output of style layers from our model
style_targets = style_transfer(style_image)['style']
# Process content image and get output of content layers from our model
content_targets = style_transfer(content_image)['content']
image = tf.Variable(content_image)
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-3)
style_weight = 0.5
content_weight = 10000
total_variation_weight = 10


def train_step(image):
    with tf.GradientTape() as tape:
        outputs = style_transfer(image)
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n(
            [tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers
        content_loss = tf.add_n(
            [tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        loss += total_variation_weight * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


epochs = 100
steps_per_epoch = 100
start = time.time()
step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')
    tensor_to_image(image).save('result_' + str(n) + '.png')
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end - start))
