/**
 * A Generic class to serialized and Unserialized classes (models, transformers,
 * or any operator)
 */

import { tf } from './shared/globals'
export default class Serialize {
  public name = 'Serialize' // default name for all inherited class

  /**
   * Serialize all [inherited] class property into
   * a json string
   * @returns Json string
   */
  public toJson(): string | Promise<string> {
    const thisCopy: any = Object.assign({}, this)
    for (const key of Object.keys(thisCopy)) {
      let value = thisCopy[key]
      if (value instanceof tf.Tensor) {
        thisCopy[key] = value.arraySync()
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

    const copyThis: any = Object.assign({}, this)
    for (let key of Object.keys(this)) {
      let value = copyThis[key]
      if (value instanceof tf.Tensor) {
        jsonClass[key] = tf.tensor(jsonClass[key])
      }
    }

    return Object.assign(this, jsonClass) as this
  }
}
