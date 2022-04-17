import { optimizer, initializer, getLoss } from '../utils'
import { tf } from '../shared/globals'
import { OneHotEncoder } from '../preprocessing/oneHotEncoder'

function getModelWeight(
  model: tf.Sequential
): Promise<tf.RecursiveArray<number>> {
  return Promise.all(model.getWeights().map((weight) => weight.array()))
}

export async function toJSON(
  classConstructor: any,
  classifierJson: any
): Promise<string> {
  const modelConfig = classConstructor.model.getConfig()
  const modelWeight = await getModelWeight(classConstructor.model)
  classifierJson.model = {
    config: modelConfig,
    weight: modelWeight
  }

  if (classConstructor.denseLayerArgs.kernelInitializer) {
    const initializerName =
      classConstructor.denseLayerArgs.kernelInitializer.constructor.name
    classifierJson.denseLayerArgs.kernelInitializer = initializerName
  }
  if (classConstructor.denseLayerArgs.biasInitializer) {
    const biasName =
      classConstructor.denseLayerArgs.biasInitializer.constructor.name
    classifierJson.denseLayerArgs.biasInitializer = biasName
  }
  // set optimizer
  classifierJson.modelCompileArgs.optimizer =
    classConstructor.model.optimizer.getConfig()
  return JSON.stringify(classifierJson)
}

export function fromJson(classConstructor: any, model: string) {
  let jsonClass = JSON.parse(model)
  if (jsonClass.name != classConstructor.name) {
    throw new Error(
      `wrong json values for ${classConstructor.name} constructor`
    )
  }

  const jsonModel = tf.Sequential.fromConfig(
    tf.Sequential,
    jsonClass.model.config
  ) as tf.Sequential
  const jsonOpt = optimizer(jsonClass.optimizerType)
  const optim = Object.assign(jsonOpt, jsonClass.modelCompileArgs.optimizer)
  const loss = getLoss(jsonClass.lossType)
  jsonClass.modelCompileArgs = {
    ...jsonClass.modelCompileArgs,
    optimizer: optim,
    loss: loss
  }

  jsonModel.compile(jsonClass.modelCompileArgs)
  const weights = []
  for (const weight of jsonClass.model.weight) {
    weights.push(tf.tensor(weight))
  }
  jsonModel.setWeights(weights)
  jsonClass.model = jsonModel

  // if call back create callback
  // default usecase is set to EarlyStop
  // might get complex for custom callback
  if (jsonClass.modelFitArgs.callbacks) {
    let jsonCallback = tf.callbacks.earlyStopping()
    let modelFitArgs = jsonClass.modelFitArgs
    jsonCallback = Object.assign(jsonCallback, modelFitArgs.callbacks[0])
    modelFitArgs.callbacks = [jsonCallback]
  }

  if (jsonClass.denseLayerArgs.kernelInitializer) {
    let initializerName = jsonClass.denseLayerArgs.kernelInitializer
    jsonClass.denseLayerArgs.kernelInitializer = initializer(initializerName)
  }
  if (jsonClass.denseLayerArgs.biasInitializer) {
    let biasName = jsonClass.denseLayerArgs.biasInitializer
    jsonClass.denseLayerArgs.biasInitializer = initializer(biasName)
  }

  if (jsonClass.oneHot) {
    let jsonOneHotEncoder = new OneHotEncoder()
    jsonClass.oneHot = Object.assign(jsonOneHotEncoder, jsonClass.oneHot)
  }
  return Object.assign(classConstructor, jsonClass)
}
