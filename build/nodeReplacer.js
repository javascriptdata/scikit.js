exports.default = function exampleReplacer({
  orig
  // file
}) {
  if (orig.includes('/shared/globals'))
    return orig.replace('/shared/globals', '/shared-node/globals')
  return orig
}
