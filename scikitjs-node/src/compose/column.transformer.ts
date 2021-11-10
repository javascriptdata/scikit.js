import { concat, Tensor2D } from '@tensorflow/tfjs-core'
import { dfd } from '../globals'
import { Scikit1D, Scikit2D, Transformer } from '../types'

// When you pass a single string or int, it "pulls" a 1D column
type Selection = string | string[] | number[] | number
type TransformerOrString = Transformer | 'drop' | 'passthrough'
type SingleTransformation = [string, Transformer, Selection]
type TransformerTriple = Array<SingleTransformation>

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isStringArray(arr: any): arr is string[] {
  return Array.isArray(arr) && typeof arr[0] === 'string'
}

export interface ColumnTransformerParams {
  transformers?: TransformerTriple
  remainder?: TransformerOrString
}

export default class ColumnTransformer {
  transformers: TransformerTriple
  remainder: TransformerOrString

  constructor({
    transformers = [],
    remainder = 'drop'
  }: ColumnTransformerParams = {}) {
    this.transformers = transformers
    this.remainder = remainder
  }

  getColumns(X: dfd.DataFrame, selectedColumns: Selection): Tensor2D {
    if (isStringArray(selectedColumns)) {
      return X.loc({ columns: selectedColumns }).tensor as Tensor2D
    }
    if (Array.isArray(selectedColumns)) {
      return X.iloc({ columns: selectedColumns }).tensor as Tensor2D
    }
    if (typeof selectedColumns === 'string') {
      return X[selectedColumns].tensor
    }
    return X.iloc({ columns: [selectedColumns] }).tensor as Tensor2D
  }

  fit(X: Scikit2D, y?: Scikit1D) {
    const newDf = X instanceof dfd.DataFrame ? X : new dfd.DataFrame(X)

    for (let i = 0; i < this.transformers.length; i++) {
      let [, curTransform, selection] = this.transformers[i]

      let subsetX = this.getColumns(newDf, selection)
      curTransform.fit(subsetX, y)
    }
    return this
  }

  transform(X: Scikit2D, y?: Scikit1D) {
    const newDf = X instanceof dfd.DataFrame ? X : new dfd.DataFrame(X)

    let output = []
    for (let i = 0; i < this.transformers.length; i++) {
      let [, curTransform, selection] = this.transformers[i]

      let subsetX = this.getColumns(newDf, selection)

      output.push(curTransform.transform(subsetX, y))
    }
    return concat(output, 1)
  }

  fitTransform(X: Scikit2D, y?: Scikit1D) {
    const newDf = X instanceof dfd.DataFrame ? X : new dfd.DataFrame(X)

    let output = []
    for (let i = 0; i < this.transformers.length; i++) {
      let [, curTransform, selection] = this.transformers[i]

      let subsetX = this.getColumns(newDf, selection)

      output.push(curTransform.fitTransform(subsetX, y))
    }
    return concat(output, 1)
  }
}
