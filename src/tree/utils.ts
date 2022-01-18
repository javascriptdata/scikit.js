import { TypedArray, int } from '../types'

export function validateX(X: number[][] | TypedArray[]) {
  if (X.length === 0) {
    throw new Error(
      `X can not be empty, but it has a length of 0. It is ${X}.`
    )
  }
  for (let i = 0; i < X.length; i++) {
    let curRow = X[i]
    if (curRow.length === 0) {
      throw new Error(
        `Rows in X can not be empty, but row ${i} in X is ${curRow}.`
      )
    }
    for (let j = 0; j < curRow.length; j++) {
      if (typeof curRow[j] !== 'number' || !Number.isFinite(curRow[j])) {
        throw new Error(
          `X must contain finite non-NaN numbers, but the element at X[${i}][${j}] is ${curRow[j]}`
        )
      }
    }
  }
}

export function validateY(y: any[] | TypedArray) {
  if (y.length === 0) {
    throw new Error(
      `y can not be empty, but it has a length of 0. It is ${y}.`
    )
  }
  for (let i = 0; i < y.length; i++) {
    let curVal = y[i]
    if (!Number.isSafeInteger(curVal)) {
      throw new Error(
        `Some y values are not an integer. Found ${curVal} but must be an integer only`
      )
    }
  }
}
