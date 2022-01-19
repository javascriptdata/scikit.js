import { tf } from '../shared/globals'
type Tensor = tf.Tensor
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

type MakeRegressionOutput = [Tensor, Tensor] | [Tensor, Tensor, Tensor]

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
  const numberInformative = Math.min(nFeatures, nInformative)

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

  const model = tf.randomNormal([numberInformative, nTargets]).mul(100)
  const zeros = tf.zeros([nFeatures - numberInformative, nTargets])
  const groundTruth = tf.concat([model, zeros])
  let Y = X.dot(groundTruth).add(bias)

  // Add noise
  if (noise > 0) {
    Y = Y.add(tf.randomNormal(Y.shape, undefined, noise))
  }

  // Randomly permute samples and features
  if (shuffle) {
    const randomTen = tf.util.createShuffledIndices(nSamples)

    X = X.gather(randomTen)
  }

  Y = tf.squeeze(Y)

  if (coef) {
    return [X, Y, tf.squeeze(groundTruth)]
  }
  return [X, Y]
}

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
}: MakeLowRankMatrixInput = {}): tf.Tensor2D => {
  let n = Math.min(nSamples, nFeatures)

  // Random (ortho normal) vectors
  let [u] = tf.linalg.qr(tf.randomNormal([nSamples, n]))
  let [v] = tf.linalg.qr(tf.randomNormal([nFeatures, n]))

  // Index of the singular values
  let singularIndex = tf.linspace(0, n - 1, n)

  // Build the singular profile by assembling signal and noise components
  const singularIndexByRank = singularIndex.div(effectiveRank)
  const lowRank = tf
    .exp(singularIndexByRank.square().neg())
    .mul(1 - tailStrength)
  let tail = tf.exp(singularIndexByRank.mul(-0.1)).mul(tailStrength)

  let s = tf.diag(lowRank.add(tail))

  return u.dot(s).dot(v.transpose()) as tf.Tensor2D
}
