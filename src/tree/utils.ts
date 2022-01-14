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
