import numpy as np
import cv2
import tensorflow as tf
import argparse
import time
from PIL import Image

import tf1st

if __name__ == "__main__":
  # Argument parse
  parser = argparse.ArgumentParser(description='Neural Style Transfer with OpenCV and Tensorflow')
  parser.add_argument('--input-image', default="./images/federer.jpg", type=str, help='image to style')
  parser.add_argument('--style-image', default="./images/vangogh.jpg", type=str, help='styling image')
  parser.add_argument('--content-weight', default=1000, type=float, help='weight of the content image')
  parser.add_argument('--style-weight', default=0.01, type=float, help='weight of the styling image')
  parser.add_argument('--iterations', default=1000, type=int, help='number of iterations')
  parser.add_argument('--result-image', default="./images/result.jpg", type=str, help='resulting image')
  args = parser.parse_args()

  # Enable eager execution for tensorflow
  tf.enable_eager_execution()
  print("Eager execution: {}".format(tf.executing_eagerly()))

  model = tf1st.get_model() 
  for layer in model.layers:
    layer.trainable = False
  
  # Get the style and content feature representations (from our specified intermediate layers) 
  style_features, content_features = tf1st.get_feature_representations(model, args.input_image, args.style_image)
  gram_style_features = [tf1st.gram_matrix(style_feature) for style_feature in style_features]
  
  # Set initial image
  init_image = tf1st.load_and_process_img(args.input_image)
  init_image = tf.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

  # Store our best result
  best_loss, best_img = float('inf'), None
  
  # Create a nice config 
  loss_weights = (args.style_weight, args.content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }

  # For displaying
  num_rows = 2
  num_cols = 5
  display_interval = args.iterations/(num_rows*num_cols)
  start_time = time.time()
  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  
  imgs = []
  for i in range(args.iterations):
    grads, all_loss = tf1st.compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time() 
    
    if loss < best_loss:
      # Update best loss and best image from total loss. 
      best_loss = loss
      best_img = tf1st.deprocess_img(init_image.numpy())
    
    start_time = time.time()  
    # Use the .numpy() method to get the concrete numpy array
    plot_img = init_image.numpy()
    plot_img = tf1st.deprocess_img(plot_img)
    imgs.append(plot_img)
    final_img = cv2.cvtColor(np.array(Image.fromarray(plot_img)), cv2.COLOR_BGR2RGB)
    cv2.imshow('Actual Styled Image', final_img)
    cv2.imwrite(args.result_image, final_img)
    cv2.waitKey(1)
    print('Iteration: {}'.format(i))        
    print('Total loss: {:.4e}, ' 
          'style loss: {:.4e}, '
          'content loss: {:.4e}, '
          'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    
  time.sleep(5)
  print('Done') 

