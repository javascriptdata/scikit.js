import { DummyClassifier } from '../dummy/dummyClassifier'
import { DummyRegressor } from '../dummy/dummyRegressor'
import { LogisticRegression } from '../linear_model/logisticRegression'
import { RidgeRegression } from '../linear_model/ridgeRegression'
import { LinearRegression } from '../linear_model/linearRegression'
import { LassoRegression } from '../linear_model/lassoRegression'
import { ElasticNet } from '../linear_model/elasticNet'
import { LabelEncoder } from '../preprocessing/labelEncoder'
import { SimpleImputer } from '../impute/simpleImputer'
import { tf } from '../shared/globals'
import { MinMaxScaler } from '../preprocessing/minMaxScaler'

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
    case 'lassoregression':
      return new LassoRegression().fromJson(serialJson)
    case 'elasticnet':
      return new ElasticNet().fromJson(serialJson)
    case 'simpleimputer':
      return new SimpleImputer().fromJson(serialJson)
    case 'minmaxscaler':
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
      const estimators = getEstimator(
        estimatorName,
        jsonEstimatorOrStep[i][1]
      )
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
