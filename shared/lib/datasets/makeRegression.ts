import * as tf from '@tensorflow/tfjs'
import { Scikit1D, Scikit2D } from 'lib'
import { dfd } from '../../globals'
import { random } from 'mathjs'

interface MakeRegressionInput {
  nSamples?: number
  nFeatures?: number
  nInformative?: number
  nTargets?: number
  bias?: number
  noise?: number
  shuffle?: boolean
}

type MakeRegressionOutput = [Scikit2D, Scikit1D]

export const makeRegression = ({
  nSamples = 100,
  nFeatures = 100,
  nInformative = 10,
  nTargets = 1,
  noise = 1,
  shuffle = false
}: MakeRegressionInput): MakeRegressionOutput => {
  // const numberInformative = math.min(nFeatures, nInformative)

  // Randomly generate a well conditioned input set
  let X: Scikit2D = tf.randomNormal([nSamples, nFeatures])

  X.print()

  // Generate a ground truth model with only n_informative features being non
  // zeros (the other features are not correlated to y and should be ignored
  // by a sparsifying regularizers such as L1 or elastic net)
  let groundTruth = tf.zeros([nFeatures, nTargets])

  // Randomly permute samples and features
  if (shuffle) {
    const randomTen = tf.util.createShuffledIndices(nSamples)

    console.log(Array(randomTen))

    X.gather(Array.from(randomTen))

    X.print()
  }

  return [X, tf.tensor1d([1, 2, 3])]
}
