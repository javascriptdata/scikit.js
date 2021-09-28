import { ArrayType1D, ArrayType2D } from "types";


/**
* Generates an array of dim (row x column) with inner values set to zero
* @param row 
* @param column 
*/
export const zeros = (row: number, column: number): ArrayType1D | ArrayType2D => {
    const zeroData = [];
    for (let i = 0; i < row; i++) {
        const colData = Array(column);
        for (let j = 0; j < column; j++) {
            colData[j] = 0;
        }
        zeroData.push(colData);
    }
    return zeroData;
}


/**
 * Checks if array is 1D
 * @param arr The array 
*/
export const is1DArray = (arr: ArrayType1D | ArrayType2D): boolean => {
    if (
        typeof arr[0] == "number" ||
        typeof arr[0] == "string" ||
        typeof arr[0] == "boolean"
    ) {
        return true;
    } else {
        return false;
    }
}