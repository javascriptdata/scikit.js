import * as tf from '@tensorflow/tfjs'
import { Tensor2D } from '@tensorflow/tfjs'
import * as math from 'mathjs'

interface MakeLowRankMatrixInput {
  nSamples?: number
  nFeatures?: number
  effectiveRank?: number
  tailStrength?: number
}

export const makeLowRankMatrix = ({
  nSamples = 100,
  nFeatures = 100,
  effectiveRank = 10,
  tailStrength = 0.5
}: MakeLowRankMatrixInput): Tensor2D => {
  let n = math.min(nSamples, nFeatures)

  // Random (ortho normal) vectors
  let [u] = tf.linalg.qr(tf.randomNormal([nSamples, n]))
  let [v] = tf.linalg.qr(tf.randomNormal([nFeatures, n]))

  // Index of the singular values
  let singularIndex = tf.linspace(0, n, n + 1)

  // Build the singular profile by assembling signal and noise components
  const singularIndexByRank = singularIndex.div(effectiveRank)
  let lowRank = tf
    .exp(
      singularIndexByRank
        .mul(singularIndexByRank)
        .mul(tf.fill(singularIndexByRank.shape, -1))
    )
    .mul(tf.fill(singularIndexByRank.shape, 1 - tailStrength))
  let tail = tf
    .exp(singularIndexByRank.mul(tf.fill(singularIndexByRank.shape, -0.1)))
    .mul(tf.fill(singularIndexByRank.shape, tailStrength))

  const identity = tf.eye(n)
  // FIX: Breaking here "Error: Operands could not be broadcast together with shapes x and y"
  let s = identity.mul(lowRank.add(tail))

  return tf.dot(tf.dot(u, s), tf.einsum('ij>ji', v)) as Tensor2D
}
