import { DummyClassifier } from '../dummy/DummyClassifier'
import { DummyRegressor } from '../dummy/DummyRegressor'
import { LogisticRegression } from '../linear_model/LogisticRegression'
import { RidgeRegression } from '../linear_model/RidgeRegression'
import { LinearRegression } from '../linear_model/LinearRegression'
import { LassoRegression } from '../linear_model/LassoRegression'
import { ElasticNet } from '../linear_model/ElasticNet'
import { LabelEncoder } from '../preprocessing/LabelEncoder'
import { SimpleImputer } from '../impute/SimpleImputer'
import { tf } from '../shared/globals'
import { MinMaxScaler } from '../preprocessing/MinMaxScaler'

function getEstimator(name: string, serialJson: string) {
  switch (name) {
    case 'DummyClassifier':
      return new DummyClassifier().fromJson(serialJson)
    case 'DummyRegressor':
      return new DummyRegressor().fromJson(serialJson)
    case 'LogisticRegression':
      return new LogisticRegression().fromJson(serialJson)
    case 'RidgeRegression':
      return new RidgeRegression().fromJson(serialJson)
    case 'LinearRegression':
      return new LinearRegression().fromJson(serialJson)
    case 'LassoRegression':
      return new LassoRegression().fromJson(serialJson)
    case 'ElasticNet':
      return new ElasticNet().fromJson(serialJson)
    case 'SimpleImputer':
      return new SimpleImputer().fromJson(serialJson)
    case 'MinMaxScaler':
      return new MinMaxScaler().fromJson(serialJson)
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
  if (jsonClass.estimators || jsonClass.steps) {
    const jsonEstimatorOrStep = jsonClass.estimators || jsonClass.steps
    for (let i = 0; i < jsonEstimatorOrStep.length; i++) {
      const estimatorName = JSON.parse(jsonEstimatorOrStep[i][1]).name
      const estimators = getEstimator(estimatorName, jsonEstimatorOrStep[i][1])
      jsonEstimatorOrStep[i][1] = Object.assign(
        estimators,
        jsonEstimatorOrStep[i][1]
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
  if (classConstructor.estimators) {
    for (const estimator of classConstructor.estimators) {
      classJson.estimators[i][1] = await estimator[1].toJson()
      i += 1
    }
  }

  if (classConstructor.steps) {
    for (const step of classConstructor.steps) {
      classJson.steps[i][1] = await step[1].toJson()
      i += 1
    }
  }

  return JSON.stringify(classJson)
}
