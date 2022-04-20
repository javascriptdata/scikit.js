let esbuild = require('esbuild')
esbuild
  .build({
    entryPoints: ['src/index.test.ts'],
    bundle: true,
    platform: 'browser',
    format: 'iife',
    outfile: 'out.js'
  })
  .catch(() => process.exit(1))
