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

function toTableParameter(obj, bigObj) {
  if (obj?.comment?.shortText) {
    return `${obj.name} | ${getTypeName(obj.type, bigObj)} | ${
      obj.comment.shortText.replace(/\n/g, ' ') +
      (obj?.comment?.text?.replace(/\n/g, '<br />') || '')
    }`
  }
  return `${obj.name} | ${getTypeName(obj.type, bigObj)}`
}

function generateTable(parameters, bigObj, isObject) {
  if (!parameters) {
    return ''
  }
  if (parameters.length === 0) {
    return ''
  }
  let noDescriptionHeader = `#### ${isObject ? 'Object ' : ''} Parameters

Name | Type
:----| :----`

  let someDescriptionHeader = `#### ${isObject ? 'Object ' : ''} Parameters

Name | Type | Description
:----| :---- | :---------`

  let hasDescription =
    parameters.map((el) => el?.comment?.shortText).filter((el) => Boolean(el))
      .length > 0

  return `
${hasDescription ? someDescriptionHeader : noDescriptionHeader}
${parameters.map((el) => toTableParameter(el, bigObj)).join('\n')}
  `
}

function generateMarkdownFromMethod(obj, className, bigObj) {
  let signature = obj.signatures[0]
  let parameters = signature.parameters
  let md = `\`\`\`typescript\n${obj.name}(${parameters
    ?.map((el) => `${el.name}${el?.flags?.isOptional ? '?' : ''}`)
    ?.join(', ')}): ${
    signature.type.name === 'default'
      ? className
      : getTypeName(signature.type, bigObj)
  }\n\`\`\`\n\n${signature?.comment?.shortText || ''}\n${generateTable(
    parameters,
    bigObj
  )}
  `
  return md
}

function getInterfaceForClass(jsonClass, bigObj) {
  let constructor = jsonClass.children.filter(
    (el) => el.kindString === 'Constructor'
  )[0]

  // Generate function call

  const signatures = constructor?.signatures
  const sig = signatures && signatures[0]
  const parameters = sig?.parameters

  const param = parameters && parameters[0]
  const type = param?.type
  if (type) {
    let interface = bigObj.children.filter(
      (el) => el.id === type.id && el.kindString === 'Interface'
    )[0]

    return interface
  }
  return undefined
}

function generateConstructor(jsonClass, bigObj) {
  let constructor = jsonClass.children.filter(
    (el) => el.kindString === 'Constructor'
  )[0]

  let interface = getInterfaceForClass(jsonClass, bigObj)
  const signatures = constructor?.signatures
  const sig = signatures && signatures[0]
  const parameters = sig?.parameters
  const noArgs = !parameters
  let constructorInvocation = `\`\`\`js\nnew ${jsonClass.name}(${
    noArgs ? '' : '{ object }'
  })\n\`\`\``

  if (interface && interface.children) {
    constructorInvocation = `\`\`\`js\nnew ${
      jsonClass.name
    }({ ${interface.children
      ?.map((el) => `${el.name}${el?.flags?.isOptional ? '?' : ''}`)
      ?.join(', ')}})\n\`\`\``

    return `### Constructor
${constructorInvocation}
${noArgs ? '' : generateTable(interface.children, {}, true)}`
  }

  return `### Constructor
${constructorInvocation}
${
  noArgs ? '' : generateMarkdownFromMethod(constructor, jsonClass.name, bigObj)
}
    `
}

function generateAllMethods(jsonClass) {
  return `
### Methods
${jsonClass.children
  .filter((el) => el.kindString === 'Method' && el?.flags?.isPublic)
  .map((el) => generateMarkdownFromMethod(el, jsonClass.name))
  .join('\n---\n')}
`
}

function readableStringsForCommonTypes({ id, name }, bigObj) {
  if (name === 'Scikit1D') {
    return 'number[] \\| string[] \\| boolean[] \\| TypedArray \\| Tensor1D \\| Series'
  }
  if (name === 'Scikit2D') {
    return '(number \\| string \\| boolean)[][] \\| TypedArray[] \\| Tensor2D \\| Dataframe'
  }
  if (name === 'dfd.DataFrame') {
    return 'DataFrame'
  }

  // if (id && bigObj && bigObj.children) {
  //   let interface = bigObj.children.filter(
  //     (el) => el.id === id && el.kindString === 'Interface'
  //   )[0]

  //   if (interface && interface.children) {
  //     return generateTable(interface.children)
  //   }
  // }

  return name
}

function getTypeName(val, bigObj) {
  if (!val) {
    return ''
  }

  const { id, type, name, value, elementType, types, typeArguments } = val
  if (type === 'reference' && name === 'Promise') {
    return `Promise<${getTypeName(typeArguments[0], bigObj)}>`
  }
  if (type === 'reference' || type === 'intrinsic') {
    return readableStringsForCommonTypes({ id, name }, bigObj)
  }
  if (type === 'literal') {
    if (typeof value === 'string') {
      return `"${value}"`
    }
    return value
  }

  if (type === 'array') {
    let typeName = getTypeName(elementType, bigObj)
    if (typeName.includes('|')) {
      return `(${typeName})[]`
    }
    return `${typeName}[]`
  }
  if (type === 'union') {
    return types.map((el) => getTypeName(el, bigObj)).join(' \\| ')
  }

  return ''
}

function generateProperties(jsonClass, bigObj) {
  let interface = getInterfaceForClass(jsonClass, bigObj)
  let allConstructorArgs = []
  if (interface && interface.children) {
    allConstructorArgs = interface.children.map((el) => el.name)
  }
  let properties = jsonClass.children.filter(
    (el) =>
      el.kindString === 'Property' && !allConstructorArgs.includes(el.name)
  )
  let propertiesText = properties
    .map((el) => {
      return `**\`${el.name}\`**: ${getTypeName(el.type, bigObj)}\n\n${
        (el?.comment?.shortText || '') + (el?.comment?.text || '')
      }`
    })
    .join('\n\n')
  return `
### Properties

${
  properties.length === 0
    ? `The only properties are the arguments defined above.\n`
    : `All of the constructor arguments above are class properties as well as \n`
}

${propertiesText}
  `
}

function writeClass(jsonClass, bigObj) {
  return `
## ${jsonClass.name}

${getText(jsonClass.comment)}
${generateConstructor(jsonClass, bigObj)}
${generateProperties(jsonClass, bigObj)}
${generateAllMethods(jsonClass)}

`
}

function writeFunction(jsonClass, bigObj) {
  let signature = {}
  if (jsonClass && jsonClass.signatures && jsonClass.signatures[0]) {
    signature = jsonClass.signatures[0]
  }
  return `
## ${jsonClass.name}

${getText(signature.comment)}

#### Signature

\`\`\`typescript
${jsonClass.name}(): ${getTypeName(signature.type, bigObj)}
\`\`\`

`
}

function doTheWholeThing(bigObj) {
  let result = bigObj.children
    .filter((el) => el.kindString === 'Class' || el.kindString === 'Function')
    .map((el) =>
      el.kindString === 'Class'
        ? writeClass(el, bigObj)
        : writeFunction(el, bigObj)
    )
    .join('\n')

  return `---
sidebar_position: 2
---

# API

${result}
`
}

let myObj = require('./out.json')
console.log(doTheWholeThing(myObj))
