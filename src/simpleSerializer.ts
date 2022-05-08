import { tf } from './shared/globals'
import { encode, decode } from 'base64-arraybuffer'
const EstimatorList = [
  'KNeighborsRegressor',
  'LinearRegression',
  'LassoRegression',
  'RidgeRegression',
  'ElasticNet',
  'LogisticRegression',
  'DummyRegressor',
  'DummyClassifier',
  'MinMaxScaler',
  'StandardScaler',
  'MaxAbsScaler',
  'SimpleImputer',
  'OneHotEncoder',
  'LabelEncoder',
  'OrdinalEncoder',
  'Normalizer',
  'Pipeline',
  'ColumnTransformer',
  'RobustScaler',
  'KMeans',
  'VotingRegressor',
  'VotingClassifier',
  'LinearSVC',
  'LinearSVR',
  'GaussianNB',
  'DecisionTreeClassifier',
  'DecisionTreeRegressor',
  'ClassificationCriterion',
  'RegressionCriterion',
  'Splitter',
  'DecisionTreeBase',
  'DecisionTree'
]

/**
 * 1. Make a list called EstimatorList
 * 2. Do a dynamic import here
 */

class JSONHandler {
  savedArtifacts: any
  constructor(artifacts?: any) {
    this.savedArtifacts = artifacts || null
  }

  async save(artifacts: any) {
    // Base 64 encoding
    artifacts.weightData = encode(artifacts.weightData)
    this.savedArtifacts = artifacts
    return {
      modelArtifactsInfo: {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: JSON.stringify(artifacts.modelTopology).length,
        weightSpecsBytes: JSON.stringify(artifacts.weightSpecs).length,
        weightDataBytes: artifacts.weightData.byteLength
      }
    }
  }

  async load() {
    // Base64 decode
    this.savedArtifacts.weightData = decode(this.savedArtifacts.weightData)
    return this.savedArtifacts
  }
}

export async function toObjectInner(
  val: any,
  ignoreKeys: string[] = []
): Promise<any> {
  if (['number', 'string', 'undefined', 'boolean'].includes(typeof val)) {
    return val
  }

  if (typeof val === 'function') {
    console.warn(
      `warning: Serializing function ${val}. Not going to be able to deserialize this later.`
    )
    if (val.name) {
      return val.name
    }
  }

  if (typeof val === 'object') {
    // Null case
    if (val === null) {
      return null
    }
    // Array case
    if (Array.isArray(val)) {
      return await Promise.all(
        val.map(async (el) => await toObjectInner(el, ignoreKeys))
      )
    }

    // Serialize a Tensor
    if (val instanceof tf.Tensor) {
      return {
        name: 'Tensor',
        value: val.arraySync()
      }
    }

    // Int32Array serialization. Used for DecisionTrees
    if (val instanceof Int32Array) {
      return {
        name: 'Int32Array',
        value: Array.from(val)
      }
    }

    // The tf object
    if (val.ENV && val.AdadeltaOptimizer && val.version) {
      return {
        name: 'TF',
        version: val.version.tfjs
      }
    }

    // tf.layers model
    if (val instanceof tf.Sequential) {
      let mem = new JSONHandler()
      await val.save(mem as any)
      return {
        name: 'Sequential',
        artifacts: mem.savedArtifacts
      }
    }

    // Generic object case / class case
    let response: any = {}
    for (let key of Object.keys(val)) {
      // Ignore all the keys that we choose to
      if (ignoreKeys.includes(key)) {
        continue
      }
      // Ignore any function when we serialize
      // if (typeof val[key] === 'function') {
      //   continue
      // }
      response[key] = await toObjectInner(val[key], ignoreKeys)
    }
    return response
  }
}

export async function fromObjectInner(val: any): Promise<any> {
  // Ignores all types that aren't objects
  if (typeof val !== 'object') {
    return val
  }

  // Null case
  if (val === null) {
    return null
  }

  // Make a Tensor
  if (val.name === 'Tensor') {
    return tf.tensor(val.value)
  }

  if (val.name === 'Sequential') {
    let newMem = new JSONHandler(val.artifacts)
    return await tf.loadLayersModel(newMem as any)
  }

  if (val.name === 'Int32Array') {
    return new Int32Array(val.value)
  }

  // Array case
  if (Array.isArray(val)) {
    return await Promise.all(val.map(async (el) => await fromObjectInner(el)))
  }

  // Generic object case
  for (let key of Object.keys(val)) {
    val[key] = await fromObjectInner(val[key])
  }

  // Make a model
  if (EstimatorList.includes(val.name)) {
    // Do dynamic import to avoid circular dependency tree
    // Every class extends this class and therefor it
    // can't import those classes in here
    let module = await import('./index')
    let model = (module as any)[val.name]

    let resultObj = new model(val)
    for (let key of Object.keys(val)) {
      resultObj[key] = val[key]
    }
    return resultObj
  }

  return val
}

export async function fromObject(val: any): Promise<any> {
  try {
    return await fromObjectInner(val)
  } catch (e) {
    console.error(e)
  }
}

export async function fromJSON(val: string): Promise<any> {
  return await fromObject(JSON.parse(val))
}

let ignoredKeysForSGDRegressor = [
  'modelCompileArgs',
  'modelFitArgs',
  'denseLayerArgs'
]

export class Serialize {
  async toObject(): Promise<any> {
    try {
      return await toObjectInner(this, ignoredKeysForSGDRegressor)
    } catch (e) {
      console.error(e)
    }
  }

  async toJSON(): Promise<string> {
    return JSON.stringify(await this.toObject())
  }
}
