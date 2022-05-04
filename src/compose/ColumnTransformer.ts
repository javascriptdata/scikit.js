import { DataFrameInterface, Scikit1D, Transformer } from '../types'
import { isDataFrameInterface } from '../typesUtils'
import { Serialize } from '../simpleSerializer'
import { tf } from '../shared/globals'
/*
Next steps:
1. Support 'passthrough' and 'drop' and estimator for remainder (also in transformer list)
2. Pass next 5 tests in scikit-learn
*/

// When you pass a single string or int, it "pulls" a 1D column
type Selection = string | string[] | number[] | number
type SingleTransformation = [string, Transformer, Selection]
type TransformerTriple = Array<SingleTransformation>

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isStringArray(arr: any): arr is string[] {
  return Array.isArray(arr) && typeof arr[0] === 'string'
}

/**
 * The parameters for the Column Transormer
 */
export interface ColumnTransformerParams {
  /**
   * A list of transformations. Every element is itself a list [name, Transformer, Selection]. **default = []**
   */
  transformers?: TransformerTriple
  /**
   * What should we do with the remainder columns? Possible values for remainder are a Transformer that
   * will be applied to all remaining columns. It can also be 'passthrough' which simply passes the columns
   * untouched through this, or 'drop', which drops all untransformed columns. **default = "drop"**
   */
  remainder?: Transformer | 'drop' | 'passthrough'
}

/**
 * The ColumnTransformer transformers a 2D matrix of mixed types, with possibly missing values
 * into a 2DMatrix that is ready to be put into a machine learning model. Usually this class does
 * the heavy lifting associated with imputing missing data, one hot encoding categorical variables,
 * and any other preprocessing steps that are deemed necessary (standard scaling, etc).
 *
 * @example
 * ```typescript
    const X = [
      [2, 2],
      [2, 3],
      [0, NaN],
      [2, 0]
    ]

    const transformer = new ColumnTransformer({
      transformers: [
        ['minmax', new MinMaxScaler(), [0]],
        ['simpleImpute', new SimpleImputer({ strategy: 'median' }), [1]]
      ]
    })

    let result = transformer.fitTransform(X)
    const expected = [
      [1, 2],
      [1, 3],
      [0, 2],
      [1, 0]
    ]
 * ```
 */
export class ColumnTransformer extends Serialize {
  transformers: TransformerTriple
  remainder: Transformer | 'drop' | 'passthrough'

  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'ColumnTransformer'

  constructor({
    transformers = [],
    remainder = 'drop'
  }: ColumnTransformerParams = {}) {
    this.transformers = transformers
    this.remainder = remainder
  }

  public fit(X: tf.Tensor2D | DataFrameInterface, y?: Scikit1D) {
    for (let i = 0; i < this.transformers.length; i++) {
      let [, curTransform, selection] = this.transformers[i]

      let subsetX = this.getColumns(X, selection)
      curTransform.fit(subsetX, y)
    }
    return this
  }

  public transform(X: tf.Tensor2D | DataFrameInterface, y?: Scikit1D) {
    let output = []
    for (let i = 0; i < this.transformers.length; i++) {
      let [, curTransform, selection] = this.transformers[i]

      let subsetX = this.getColumns(X, selection)

      output.push(curTransform.transform(subsetX, y))
    }
    return tf.concat(output, 1)
  }

  public fitTransform(X: tf.Tensor2D | DataFrameInterface, y?: Scikit1D) {
    let output = []
    for (let i = 0; i < this.transformers.length; i++) {
      let [, curTransform, selection] = this.transformers[i]

      let subsetX = this.getColumns(X, selection)

      output.push(curTransform.fitTransform(subsetX, y))
    }
    return tf.concat(output, 1)
  }

  getColumns(
    X: DataFrameInterface | tf.Tensor2D,
    selectedColumns: Selection
  ): tf.Tensor2D {
    if (isDataFrameInterface(X)) {
      if (isStringArray(selectedColumns)) {
        return X.loc({ columns: selectedColumns })
          .tensor as unknown as tf.Tensor2D
      }
      if (Array.isArray(selectedColumns)) {
        return X.iloc({ columns: selectedColumns })
          .tensor as unknown as tf.Tensor2D
      }
      if (typeof selectedColumns === 'string') {
        return X[selectedColumns].tensor
      }
      return X.iloc({ columns: [selectedColumns] })
        .tensor as unknown as tf.Tensor2D
    } else {
      if (
        isStringArray(selectedColumns) ||
        typeof selectedColumns === 'string'
      ) {
        throw new Error(
          "Can't pass string selected columns when not a DataFrame"
        )
      }
      if (typeof selectedColumns === 'number') {
        let columns = tf.tensor1d([selectedColumns])
        return X.gather(columns, 1)
      } else {
        let columns = tf.tensor1d(selectedColumns)
        return X.gather(columns, 1)
      }
    }
  }
}
