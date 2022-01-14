import { Scikit1D, Scikit2D } from '../types'
import { min } from 'mathjs'
import { tf } from '../shared/globals'
import { makeLowRankMatrix } from './makeLowRankMatrix'

interface MakeRegressionInput {
  nSamples?: number
  nFeatures?: number
  nInformative?: number
  nTargets?: number
  bias?: number
  effectiveRank?: number | null
  tailStrength?: number
  noise?: number
  shuffle?: boolean
  coef?: boolean
}

type MakeRegressionOutput =
  | [Scikit2D, Scikit1D]
  | [Scikit2D, Scikit1D, Scikit1D]

export const makeRegression = ({
  nSamples = 100,
  nFeatures = 100,
  nInformative = 10,
  nTargets = 1,
  noise = 1,
  bias = 0,
  effectiveRank = null,
  tailStrength = 0.5,
  shuffle = false,
  coef = false
}: MakeRegressionInput = {}): MakeRegressionOutput => {
  const numberInformative = min(nFeatures, nInformative)

  let X: tf.Tensor2D
  if (effectiveRank === null) {
    // Randomly generate a well conditioned input set
    X = tf.randomNormal([nSamples, nFeatures])
  } else {
    X = makeLowRankMatrix({ nSamples, nFeatures, effectiveRank, tailStrength })
  }

  // Generate a ground truth model with only n_informative features being non
  // zeros (the other features are not correlated to y and should be ignored
  // by a sparsifying regularizers such as L1 or elastic net)
  let [groundTruthA, groundTruthB] = tf.split(
    tf.zeros([nFeatures, nTargets]),
    [numberInformative, nFeatures - numberInformative],
    0
  )
  let groundTruthRandom = tf.randomNormal([numberInformative, nTargets])
  groundTruthRandom = groundTruthRandom.mul(
    tf.fill(groundTruthRandom.shape, 100)
  )
  groundTruthA = groundTruthA.add(groundTruthRandom)

  let groundTruth = tf.concat([groundTruthA, groundTruthB])

  let Y: tf.Tensor1D = tf.einsum('i,i->', X, groundTruth).as1D()

  Y = Y.add(tf.fill(Y.shape, bias)).as1D()

  // Add noise
  if (noise > 0) {
    Y = Y.add(tf.randomNormal(Y.shape, undefined, noise)).as1D()
  }

  // Randomly permute samples and features
  if (shuffle) {
    const randomTen = tf.util.createShuffledIndices(nSamples)

    X.gather(Array.from(randomTen))
  }

  Y = tf.squeeze(Y).as1D()

  if (coef) {
    return [X, Y, tf.squeeze(groundTruth).as1D()]
  }
  return [X, Y]
}
