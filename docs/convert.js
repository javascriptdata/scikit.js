function getText(obj) {
  let example = obj?.tags
    ?.filter((el) => el.tag === 'example')
    ?.map((el) => el.text)
    ?.join('\n')

  if (example?.length > 0) {
    example = `\n#### Example ${example}\n`
  }
  return `${obj?.shortText || ''}${obj?.text || ''}${example || ''}`
}

function toTableParameter(obj) {
  return `${obj.name} | ${obj.type.name}`
}

function generateTable(parameters) {
  if (!parameters) {
    return ''
  }
  if (parameters.length === 0) {
    return ''
  }
  return `
#### Parameters

Name | Type
:----| :----
${parameters.map((el) => toTableParameter(el)).join('\n')}
  `
}

function generateMarkdownFromMethod(obj, className) {
  let signature = obj.signatures[0]
  let parameters = signature.parameters
  let md = `
**${obj.name}(${parameters
    ?.map((el) => `${el.name}${el?.flags?.isOptional ? '?' : ''}`)
    ?.join(',')})**: ${
    signature.type.name === 'default' ? className : signature.type.name
  }\n\n${signature?.comment?.shortText || ''}\n${generateTable(parameters)}
  `
  return md
}

function generateAllMethods(jsonClass) {
  return `
### Methods
${jsonClass.children
  .filter((el) => el.kindString === 'Method')
  .map((el) => generateMarkdownFromMethod(el, jsonClass.name))
  .join('\n---\n')}
`
}

function generateProperties(jsonClass) {
  let properties = jsonClass.children
    .filter((el) => el.kindString === 'Property')
    .map((el) => {
      return `**${el.name}**: ${el.type.name}\n\n${
        el?.comment?.shortText || ''
      }`
    })
    .join('\n\n')
  return `
### Properties
${properties}
  `
}

function writeClass(jsonClass) {
  return `
## ${jsonClass.name}

${getText(jsonClass.comment)}
${generateProperties(jsonClass)}
${generateAllMethods(jsonClass)}

`
}

function doTheWholeThing(bigObj) {
  let result = bigObj.children
    .filter((el) => el.kindString === 'Class')
    .map((el) => writeClass(el))
    .join('\n')

  return `

${result}
`
}

let myObj = require('./out.json')
console.log(doTheWholeThing(myObj))
