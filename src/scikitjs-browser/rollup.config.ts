import typescript from '@rollup/plugin-typescript'
import { nodeResolve } from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import { terser } from 'rollup-plugin-terser'

export default [
  {
    input: 'src/shared/index.ts',
    output: {
      file: `dist/scikit.min.js`,
      format: 'umd',
      name: 'scikit',
      esModule: false,
      exports: 'named',
      sourcemap: true
    },
    plugins: [
      typescript({ module: 'esnext' }),
      nodeResolve(),
      commonjs(),
      terser()
    ]
  },
  {
    input: 'src/shared/index.ts',
    plugins: [typescript({ module: 'esnext' })],
    output: {
      dir: 'dist',
      format: 'esm',
      exports: 'named',
      sourcemap: true
    }
  }
]
