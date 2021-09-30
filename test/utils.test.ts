import * as tf from '@tensorflow/tfjs-node';

/**
 * Check that if two tensor are of same shape
 * @param tensor1
 * @param tensor2
 * @returns
 */
export const shapeEqual = (
  tensor1: tf.Tensor,
  tensor2: tf.Tensor
): boolean => {
  const shape1 = tensor1.shape;
  const shape2 = tensor2.shape;
  if (shape1.length != shape2.length) {
    return false;
  }
  for (let i = 0; i < shape1.length; i++) {
    if (shape1[i] !== shape2[i]) {
      return false;
    }
  }
  return true;
};

/**
 * Check that two tensors are equal to within some additive tolerance.
 * @param tensor1
 * @param tensor2
 * @param
 */
export const tensorEqual = (
  tensor1: tf.Tensor,
  tensor2: tf.Tensor,
  tol = 0
): boolean => {
  if (!shapeEqual(tensor1, tensor2)) {
    throw new Error('tensor1 and tensor2 not of same shape');
  }
  return Boolean(
    tf.lessEqual(tf.max(tf.abs(tf.sub(tensor1, tensor2))), tol).dataSync()[0]
  );
};
