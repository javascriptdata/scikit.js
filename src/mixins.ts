export class TransformerMixin {
  [x: string]: any
  fitTransform(X: any) {
    return this.fit(X).transform(X)
  }
}

export class PredictorMixin {
  [x: string]: any
  fitPredict(X: any) {
    return this.fit(X).predict(X)
  }
}
