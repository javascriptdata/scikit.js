import { Scikit2D, Scikit1D } from './types'
import { r2Score, accuracyScore } from './metrics/metrics'
// import Serialize from './serialize'
import { Serialize } from './simpleSerializer'
import { tf } from './shared/globals'
export class TransformerMixin extends Serialize {
  // We assume that fit and transform exist
  [x: string]: any

  public fitTransform(X: Scikit2D): tf.Tensor2D {
    return this.fit(X).transform(X)
  }
}

export class PredictorMixin {
  // We assume that fit and predict exist
  [x: string]: any

  public fitPredict(X: Scikit2D, y: Scikit1D): tf.Tensor1D {
    return this.fit(X, y).predict(X)
  }
}

export class RegressorMixin extends Serialize {
  // We assume we have a predict function
  [x: string]: any

  EstimatorType = 'regressor'
  public score(X: Scikit2D, y: Scikit1D): number {
    const yPred = this.predict(X)
    return r2Score(y, yPred)
  }
}

export class ClassifierMixin extends Serialize {
  // We assume we have a predict function
  [x: string]: any

  EstimatorType = 'classifier'
  public score(X: Scikit2D, y: Scikit1D): number {
    const yPred = this.predict(X)
    return accuracyScore(y, yPred)
  }
}

export const mixins = (baseClass: any, ...mixins: any[]): any => {
  class base extends baseClass {
    constructor(...args: any[]) {
      super(...args)
      mixins.forEach((mixin) => {
        copyProps(this, new mixin())
      })
    }
  }
  let copyProps = (target: any, source: any) => {
    // this function copies all properties and symbols, filtering out some special ones
    Object.getOwnPropertyNames(source)
      .concat((Object as any).getOwnPropertySymbols(source))
      .forEach((prop) => {
        if (
          !prop.match(
            /^(?:constructor|prototype|arguments|caller|name|bind|call|apply|toString|length)$/
          )
        )
          Object.defineProperty(
            target,
            prop,
            (Object as any).getOwnPropertyDescriptor(source, prop)
          )
      })
  }
  mixins.forEach((mixin) => {
    // outside contructor() to allow aggregation(A,B,C).staticFunction() to be called etc.
    copyProps(base.prototype, mixin.prototype)
    copyProps(base, mixin)
  })
  return base
}
