/**
 * A Generic class to serialized and Unserialized classes (models, transformers,
 * or any operator)
 */

import { tf } from './shared/globals'
import omit from 'lodash/omit'
export default class Serialize {
  public name = 'Serialize' // default name for all inherited class

  /**
   * Serialize all [inherited] class property into
   * a json string
   * @returns Json string
   */
  public toJson(): string | Promise<string> {
    const thisCopy: any = Object.assign({}, omit(this, 'tf'))
    console.log(thisCopy)
    for (const key of Object.keys(thisCopy)) {
      let value = thisCopy[key]

      if (value instanceof tf.Tensor) {
        thisCopy[key] = {
          type: 'Tensor',
          value: value.arraySync()
        }
      }
    }
    return JSON.stringify(thisCopy)
  }

  /**
   * Initialize [inherited] class from serialized
   * json string
   * @param model string
   * @returns [Inherited] Class
   */
  public fromJson(model: string) {
    let jsonClass = JSON.parse(model)
    if (jsonClass.name != this.name) {
      throw new Error(`wrong json values for ${this.name} constructor`)
    }

    for (let key of Object.keys(jsonClass)) {
      let value = jsonClass[key]
      if (typeof value === 'object' && value?.type === 'Tensor') {
        jsonClass[key] = tf.tensor(jsonClass[key].value)
      }
    }

    return Object.assign(this, jsonClass) as this
  }
}
