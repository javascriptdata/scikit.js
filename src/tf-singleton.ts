let tf: any = null

export function setBackend(tfInput: any) {
  tf = tfInput
}

export function getBackend() {
  if (tf === null) {
    throw Error(`
============================
Howdy 👋👋. Looks like you are running scikit but you haven't set a Tensorflow backend. 
To do so, simply import (or require) your tensorflow library, and call setBackend like so,

import * as tf from '@tensorflow/tfjs'
import * as sk from 'scikitjs'
sk.setBackend(tf)

That will let scikit know you wish to use a tensorflow library to perform your calculations.
============================
    `)
  }
  return tf
}
