exports.default = function exampleReplacer({
  orig
  // file
}) {
  if (orig.includes('/shared-node/globals'))
    return orig.replace('/shared-node/globals', '/shared-esm/globals')
  else if (orig.includes('/shared/globals'))
    return orig.replace('/shared/globals', '/shared-esm/globals')
  return orig
}
