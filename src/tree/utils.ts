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

export function quickSort(
  items: any[],
  left: number,
  right: number,
  key: string
) {
  let index
  if (items.length > 1) {
    index = partition(items, left, right, key) //index returned from partition
    if (left < index - 1) {
      //more elements on the left side of the pivot
      quickSort(items, left, index - 1, key)
    }
    if (index < right) {
      //more elements on the right side of the pivot
      quickSort(items, index, right, key)
    }
  }
  return items
}

function swap(items: any[], leftIndex: number, rightIndex: number) {
  let temp = items[leftIndex]
  items[leftIndex] = items[rightIndex]
  items[rightIndex] = temp
}
function partition(items: any[], left: number, right: number, key: string) {
  let pivot = items[Math.floor((right + left) / 2)] //middle element
  let i = left //left pointer
  let j = right //right pointer
  while (i <= j) {
    while (items[i][key] < pivot[key]) {
      i++
    }
    while (items[j][key] > pivot[key]) {
      j--
    }
    if (i <= j) {
      swap(items, i, j) //swapping two elements
      i++
      j--
    }
  }
  return i
}
