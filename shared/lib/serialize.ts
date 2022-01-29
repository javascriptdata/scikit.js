/**
 * A Generic class to serialized and Unserialized classes (models, transformers,
 * or any operator)
 */
export default class Serialize {
  public name = 'Serialize' // default name for all inherited class

  /**
   * Serialize all [inherited] class property into
   * a json string
   * @returns Json string
   */
  public toJson(): string {
    return JSON.stringify(this)
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

    return Object.assign(this, jsonClass)
  }
}
