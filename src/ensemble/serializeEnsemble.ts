import { DummyClassifier } from '../dummy/dummyClassifier'
import { DummyRegressor } from '../dummy/dummyRegressor'
import { LogisticRegression } from '../linear_model/logisticRegression'
import { RidgeRegression } from '../linear_model/ridgeRegression'
import { LinearRegression } from '../linear_model/linearRegression'
import { LassoRegression } from '../linear_model/lassoRegression'
import { ElasticNet } from '../linear_model/elasticNet'
import { LabelEncoder } from '../preprocessing/labelEncoder'
import { tf } from '../shared/globals'

function getEstimator(name: string, serialJson: string) {
  switch (name) {
    case 'dummyclassifier':
      return new DummyClassifier().fromJson(serialJson)
    case 'dummyregressor':
      return new DummyRegressor().fromJson(serialJson)
    case 'logisticregression':
      return new LogisticRegression().fromJson(serialJson)
    case 'ridgeregression':
      return new RidgeRegression().fromJson(serialJson)
    case 'linearregression':
      return new LinearRegression().fromJson(serialJson)
    case 'lassoregresion':
      return new LassoRegression().fromJson(serialJson)
    case 'elasticnet':
      return new ElasticNet().fromJson(serialJson)
    default:
      throw new Error(`${name} estimator not supported`)
  }
}

export function fromJson(classConstructor: any, model: string) {
  let jsonClass = JSON.parse(model)
  if (jsonClass.name != classConstructor.name) {
    throw new Error(
      `wrong json values for ${classConstructor.name} constructor`
    )
  }

  const copyThis: any = Object.assign({}, classConstructor)
  for (let key of Object.keys(classConstructor)) {
    let value = copyThis[key]
    if (value instanceof tf.Tensor) {
      jsonClass[key] = tf.tensor(jsonClass[key])
    }
  }
  // for ensembles
  if (jsonClass.estimators) {
    for (let i = 0; i < jsonClass.estimators.length; i++) {
      const estimatorName = jsonClass.estimators[i][0]
      const estimators = getEstimator(
        estimatorName,
        jsonClass.estimators[i][1]
      )
      jsonClass.estimators[i][1] = Object.assign(
        estimators,
        jsonClass.estimators[i][1]
      )
    }
  }

  if (jsonClass.le) {
    const labelEncode = new LabelEncoder()
    jsonClass.le = Object.assign(labelEncode, jsonClass.le)
  }
  return Object.assign(classConstructor, jsonClass)
}

export async function toJson(classConstructor: any, classJson: any) {
  let i = 0
  for (const estimator of classConstructor.estimators) {
    classJson.estimators[i][1] = await estimator[1].toJson()
    i += 1
  }
  return JSON.stringify(classJson)
}
