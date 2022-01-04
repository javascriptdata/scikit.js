import * as tf from '@tensorflow/tfjs'
import { Scikit1D, Scikit2D } from '../'
import * as math from 'mathjs'

interface MakeRegressionInput {
  nSamples?: number
  nFeatures?: number
  nInformative?: number
  nTargets?: number
  bias?: number
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
  shuffle = false,
  coef = false
}: MakeRegressionInput): MakeRegressionOutput => {
  const numberInformative = math.min(nFeatures, nInformative)

  // Randomly generate a well conditioned input set
  let X: Scikit2D = tf.randomNormal([nSamples, nFeatures])

  // Generate a ground truth model with only n_informative features being non
  // zeros (the other features are not correlated to y and should be ignored
  // by a sparsifying regularizers such as L1 or elastic net)
  let [groundTruthA, groundTruthB] = tf.split(
    tf.zeros([nFeatures, nTargets]),
    [numberInformative, nFeatures - numberInformative],
    0
  )
  const groundTruthRandom = tf.randomNormal([numberInformative, nTargets])
  groundTruthA = groundTruthA.add(groundTruthRandom)

  let groundTruth = tf.concat([groundTruthA, groundTruthB])

  let Y: Scikit1D = tf.einsum('i,i->', X, groundTruth).as1D()

  Y = Y.add(tf.fill(Y.shape, bias)).as1D()

  // Add noise
  if (noise > 0) {
    Y = Y.add(tf.randomNormal(Y.shape, undefined, noise)).as1D()
  }

  // Randomly permute samples and features
  if (shuffle) {
    const randomTen = tf.util.createShuffledIndices(nSamples)

    console.log(Array(randomTen))

    X.gather(Array.from(randomTen))

    X.print()
  }

  Y = tf.squeeze(Y).as1D()

  X.print()
  Y.print()

  if (coef) return [X, Y, tf.squeeze(groundTruth).as1D()]

  return [X, Y]
}

makeRegression({ nSamples: 4, nFeatures: 4, nInformative: 2, nTargets: 1, noise: 0.1 })
