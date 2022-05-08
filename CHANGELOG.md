# [1.21.0](https://github.com/javascriptdata/scikit.js/compare/v1.20.0...v1.21.0) (2022-05-08)


### Features

* not complete serialization ([7fbea07](https://github.com/javascriptdata/scikit.js/commit/7fbea07431dbd194259e44e868df74617814f6de))
* updated serialization ([ec71323](https://github.com/javascriptdata/scikit.js/commit/ec713230efbf07ac228c0291c21f5fcc0b1bd995))
* updated test ([260c134](https://github.com/javascriptdata/scikit.js/commit/260c1347d7c06c494b24b813c00c11d68c4da354))

# [1.20.0](https://github.com/javascriptdata/scikit.js/compare/v1.19.0...v1.20.0) (2022-04-26)


### Features

* updated tensorflow ([00d1863](https://github.com/javascriptdata/scikit.js/commit/00d1863e42979940810d49a3b63c0ff7fdc0c109))

# [1.19.0](https://github.com/javascriptdata/scikit.js/compare/v1.18.0...v1.19.0) (2022-04-26)


### Features

* changed lodash imports to support building on esm.sh ([3eabad9](https://github.com/javascriptdata/scikit.js/commit/3eabad9e10b9155eefdc9ae93e2c2a1b548e9496))

# [1.18.0](https://github.com/javascriptdata/scikit.js/compare/v1.17.0...v1.18.0) (2022-04-26)


### Features

* removed seedrandom in favor of inlining to help build on esm.sh ([245d49c](https://github.com/javascriptdata/scikit.js/commit/245d49c904fea6ffab9201cab58c4504b6ddfe3a))

# [1.17.0](https://github.com/javascriptdata/scikit.js/compare/v1.16.0...v1.17.0) (2022-04-21)


### Features

* added automated tests to test our code in the browser ([87e06a2](https://github.com/javascriptdata/scikit.js/commit/87e06a26dea5147f3ea22a568f558a34584706b5))
* renamed files, added to repo ([9a63da4](https://github.com/javascriptdata/scikit.js/commit/9a63da499910f0c693622f3eb56c3846672e2365))

# [1.16.0](https://github.com/javascriptdata/scikit.js/compare/v1.15.0...v1.16.0) (2022-04-19)


### Features

* fixed loadBoston calls. Need to do the others ([05c9d9a](https://github.com/javascriptdata/scikit.js/commit/05c9d9af772c1d197798ea28fdd985e92b6fc5ac))
* fixed tests ([3f6654d](https://github.com/javascriptdata/scikit.js/commit/3f6654d3aa128cfc0296cd97cb4f672cb2184bc9))
* remove data loading logic in favor of using dfd.readCSV(url) ([3251738](https://github.com/javascriptdata/scikit.js/commit/3251738e09b9e1af9a354b225033a57b1081f573))

# [1.15.0](https://github.com/javascriptdata/scikit.js/compare/v1.14.0...v1.15.0) (2022-04-18)


### Features

* remove rollup from the build process, replace with esbuild ([1f16ef8](https://github.com/javascriptdata/scikit.js/commit/1f16ef83fe291509e2dac31e4c214392013c12d6))
* updated readme ([7e70aba](https://github.com/javascriptdata/scikit.js/commit/7e70aba606c894ed1a128bf3fc4764d02952deec))

# [1.14.0](https://github.com/javascriptdata/scikit.js/compare/v1.13.0...v1.14.0) (2022-04-17)


### Features

* commented out tests ([77b6ab6](https://github.com/javascriptdata/scikit.js/commit/77b6ab64c3c59fdcba8f87f8e48a6bb77f90438b))
* commenting out svc, svr code until it can be built in browser ([dd95256](https://github.com/javascriptdata/scikit.js/commit/dd952567e6b41c5e314e99f62debae4e81b9d0f5))
* disable libsvm until we can ship to the browser ([fdc3214](https://github.com/javascriptdata/scikit.js/commit/fdc3214649ebd0a8868ce183b45e199fa7b99a0c))
* updated tests ([6938b32](https://github.com/javascriptdata/scikit.js/commit/6938b326cee279bd41700eb73bf9b2b9b4153096))

# [1.13.0](https://github.com/javascriptdata/scikit.js/compare/v1.12.0...v1.13.0) (2022-04-17)


### Features

* browser package json ([c13d968](https://github.com/javascriptdata/scikit.js/commit/c13d9687a40e70c025897d16dc1175c6812fd475))

# [1.12.0](https://github.com/javascriptdata/scikit.js/compare/v1.11.0...v1.12.0) (2022-04-17)


### Features

* only import from tensorflow and not subpackages ([f971942](https://github.com/javascriptdata/scikit.js/commit/f971942ab40f4c12ce84fb37f39c669db9b95250))

# [1.11.0](https://github.com/javascriptdata/scikit.js/compare/v1.10.0...v1.11.0) (2022-04-17)


### Features

* updated export map ([8b688bc](https://github.com/javascriptdata/scikit.js/commit/8b688bcc56d99456e1a3b40921ea3858edd6e808))

# [1.10.0](https://github.com/javascriptdata/scikit.js/compare/v1.9.0...v1.10.0) (2022-04-17)


### Features

* removed danfo as a dependency ([b8b5578](https://github.com/javascriptdata/scikit.js/commit/b8b5578bf66f6c3978bf03cb3f9b21895b2aaca7))

# [1.9.0](https://github.com/javascriptdata/scikit.js/compare/v1.8.0...v1.9.0) (2022-02-27)


### Bug Fixes

* fix ci issue base on update to serializer ([c8bf774](https://github.com/javascriptdata/scikit.js/commit/c8bf774bb23ff26ab4020b2042829db1028b2015))
* fix lint error ([6590072](https://github.com/javascriptdata/scikit.js/commit/6590072fdbe9fd4adf740fa0284f7110a10e2b31))
* fix lint errors ([e8dab99](https://github.com/javascriptdata/scikit.js/commit/e8dab99d1d5292c5e9808e91b3975ee13162c820))
* fix lint issues ([61e584b](https://github.com/javascriptdata/scikit.js/commit/61e584b2d19ce6abf57c374c3b53e14742b1ce0b))
* fix type errors ([83a78dc](https://github.com/javascriptdata/scikit.js/commit/83a78dc76f7041775014e5e6bbdef2cc4592b939))


### Features

* add custom serializer to sgdclassifier (wip) ([9c3f3dc](https://github.com/javascriptdata/scikit.js/commit/9c3f3dc1af84fede71bd0d37e2f19bee36d580fc))
* add loss and optimizer type to enable easy parsing ([08809ec](https://github.com/javascriptdata/scikit.js/commit/08809ec121056156e4bfb13d4677248f6f1ef830))
* add loss types  and initializer types ([33e1d2c](https://github.com/javascriptdata/scikit.js/commit/33e1d2ca6ee084c4ef0c76119e7af546c012f9f4))
* add more estimators and makes serializer flexible for ensembles and pipeline ([e2d319b](https://github.com/javascriptdata/scikit.js/commit/e2d319b52173e388198696a0b1499af8b70e62fa))
* add optimizer, loss and intializer caller ([8778f73](https://github.com/javascriptdata/scikit.js/commit/8778f731200ba8860133c142be3558a918d46699))
* add serializer to criterion ([8cbb737](https://github.com/javascriptdata/scikit.js/commit/8cbb7374ac81f585d10aba950a03883633e3846c))
* add serializer to criterion ([92f765e](https://github.com/javascriptdata/scikit.js/commit/92f765e6cdb2028b055e281623f3137668e0737c))
* add serializer to decision tree and update test ([83ef949](https://github.com/javascriptdata/scikit.js/commit/83ef949fb524877f49ceb8418105bf0085769f41))
* add serializer to kNeighborBase ([5314044](https://github.com/javascriptdata/scikit.js/commit/531404440a6eaebeb073bd704370bc3de5cbdfa5))
* add serializer to labelencoder ([6a1c362](https://github.com/javascriptdata/scikit.js/commit/6a1c3626914a473b6c32d5bd4038a7e517bda87a))
* add serializer to linear model base class ([9701a3e](https://github.com/javascriptdata/scikit.js/commit/9701a3e1fe1646dea7e1542d81fe8082b522f54e))
* add serializer to NaiveBayes ([84e747e](https://github.com/javascriptdata/scikit.js/commit/84e747ef27b9e0a42c5b7e56ea113abab6779cbe))
* add serializer to pipeline ([6c425e1](https://github.com/javascriptdata/scikit.js/commit/6c425e1cbb778d756ad4541c6ee252909c73df9a))
* add serializer to splitter ([763d7e1](https://github.com/javascriptdata/scikit.js/commit/763d7e13c5ef38f7331ed5fd90349c268e175647))
* add serializer to SVC AND SVR ([672328f](https://github.com/javascriptdata/scikit.js/commit/672328f048832918e950f44cd394004901323b73))
* add serializer to votingclassifier ([2a21f31](https://github.com/javascriptdata/scikit.js/commit/2a21f317f7bcffbb80b73a22652ceaf5d7038f07))
* add serializer to votingRegressor ([6d59faf](https://github.com/javascriptdata/scikit.js/commit/6d59faf2548b0e1707996a7b54f42d60bba102d0))
* allow ClassifierMixin to extends Serialize class ([48d15f1](https://github.com/javascriptdata/scikit.js/commit/48d15f1db3fa967f248fbe2f17f8bbf11d16035d))
* allow Kmeans to inherit from serializer ([6b36f4c](https://github.com/javascriptdata/scikit.js/commit/6b36f4cbc1450cc0f72201c3b3e435cb3955965f))
* implement generic class to Serialize models and transformers e.t.c ([7f617bc](https://github.com/javascriptdata/scikit.js/commit/7f617bcfd93ed6e214ee4a05ccb3eaa8d8adc808))
* implement serialize ensembles for ensemble class ([bbd9fac](https://github.com/javascriptdata/scikit.js/commit/bbd9facb6b332b7ba536b7275d028bcc918ba9af))
* make TransformerMixin and RegressorMixin extends serialize ([5a001fd](https://github.com/javascriptdata/scikit.js/commit/5a001fdbca241f916396a66aaf4d6eec0d0e50ad))
* update linear model with new args to enable easy serialization ([ec559e1](https://github.com/javascriptdata/scikit.js/commit/ec559e192e12c5b2bb8e2849932110de762e78f0))
* update serialize to easily parse tensors ([232cb62](https://github.com/javascriptdata/scikit.js/commit/232cb621ae1090b15840e7ff6f68311593c1f13e))
* update Serialize to handle serialization of tensors ([924b050](https://github.com/javascriptdata/scikit.js/commit/924b0501de07f560a78e9fac6f643d19f6a84a89))
* update serialize to return inherited class ([0733023](https://github.com/javascriptdata/scikit.js/commit/073302383228af55091e0b378aac139f3c429638))
* update serializer for sgdclassifier ([4971b3c](https://github.com/javascriptdata/scikit.js/commit/4971b3cde5773e7c2622112b4f103df004f26263))

# [1.8.0](https://github.com/javascriptdata/scikit.js/compare/v1.7.0...v1.8.0) (2022-01-28)


### Bug Fixes

* k-neighbors-regressor now supports no params ([9656d6b](https://github.com/javascriptdata/scikit.js/commit/9656d6b378db3d81fdb927ec5daaa97092b644d2))
* kd-tree index issue fix + docs ([5eba76c](https://github.com/javascriptdata/scikit.js/commit/5eba76c5a9c0daa4f884880fdb5ee2d833347f49))
* kd-tree protection copy + tfjs-core import ([5c4348d](https://github.com/javascriptdata/scikit.js/commit/5c4348d8f7a52b9c675365753d4647b41cb435f9))


### Features

* k-neighbors now lists available algorithms ([fcfcb87](https://github.com/javascriptdata/scikit.js/commit/fcfcb879ce857b110c6cd60a0e857656a7fd15eb))

# [1.7.0](https://github.com/javascriptdata/scikit.js/compare/v1.6.0...v1.7.0) (2022-01-23)


### Bug Fixes

* cross-val-score and k-fold fixes+improvements ([21a566b](https://github.com/javascriptdata/scikit.js/commit/21a566b98e6e05cab620008fd4d913eebdee495b))
* cross-val-score api improvement etc ([efe63f9](https://github.com/javascriptdata/scikit.js/commit/efe63f9683ee100182a1a6cdfc0148572aed7174))
* k-fold memory leak ([2f5529d](https://github.com/javascriptdata/scikit.js/commit/2f5529d9ba1204dfaec3c85234d83f11f453634e))


### Features

* cross-val-score and k-fold implemented ([6bc3ee3](https://github.com/javascriptdata/scikit.js/commit/6bc3ee38f118bc094ad31d44f9494308bb25b311))
* rand-utils create-rng ([553232f](https://github.com/javascriptdata/scikit.js/commit/553232f5b57db1f2834bba02cd95ed039e71b39a))

# [1.6.0](https://github.com/javascriptdata/scikit.js/compare/v1.5.0...v1.6.0) (2022-01-21)


### Features

* fixed build ([cbdbbda](https://github.com/javascriptdata/scikit.js/commit/cbdbbda59a819c0a777b0eba40263abf16a24d2c))

# [1.5.0](https://github.com/javascriptdata/scikit.js/compare/v1.4.0...v1.5.0) (2022-01-20)


### Features

* add makeRegression function ([5337ecf](https://github.com/javascriptdata/scikit.js/commit/5337ecfc6bb77fd6cfe0f278e66555b194194b10))

# [1.4.0](https://github.com/javascriptdata/scikit.js/compare/v1.3.0...v1.4.0) (2022-01-18)


### Features

* added ability for decision tree to handle negative input ([a6cf53f](https://github.com/javascriptdata/scikit.js/commit/a6cf53ff301bbf463187b4aa41519ec6c25d8d06))
* first pass at decision tree classifier ([550551e](https://github.com/javascriptdata/scikit.js/commit/550551eb0beca2effb6428368d1b9c94d27a0fb1))
* first pass at regression tree ([849469a](https://github.com/javascriptdata/scikit.js/commit/849469ab103e1e5e2c5e2a33773bf2fc4476baa9))

# [1.3.0](https://github.com/javascriptdata/scikit.js/compare/v1.2.0...v1.3.0) (2022-01-14)


### Features

* gaussian naive bayes classifier ([8174ae1](https://github.com/javascriptdata/scikit.js/commit/8174ae17806b18d5d4488fa2d26086463dc240dc))
* gaussian naive bayes classifier ([d520b1a](https://github.com/javascriptdata/scikit.js/commit/d520b1a9720ad613bcf08915c03233ce6958ebe2))

# [1.2.0](https://github.com/javascriptdata/scikit.js/compare/v1.1.0...v1.2.0) (2022-01-02)


### Features

* seeing if this package.json exports does the trick ([4a73f7c](https://github.com/javascriptdata/scikit.js/commit/4a73f7c79281d8304eec9dbb47293f03b14fc383))

# [1.1.0](https://github.com/javascriptdata/scikit.js/compare/v1.0.3...v1.1.0) (2021-12-31)


### Bug Fixes

* added fast-check dev dependency ([fe9e693](https://github.com/javascriptdata/scikit.js/commit/fe9e6939a0f0e70f55f1f2667fcbadacc2c1ca7f))
* change max length on commit message ([fe4ce57](https://github.com/javascriptdata/scikit.js/commit/fe4ce57a9a7db22f57b9a6c621e98f72ae2624db))
* change max length on commit message ([f4a8672](https://github.com/javascriptdata/scikit.js/commit/f4a86724f26776e9139dfb3bac8d1aa09a50a08d))
* commented out tests failing in test:browser ([0fe0fe1](https://github.com/javascriptdata/scikit.js/commit/0fe0fe1dba9ee2a42e7faf3e5a65d25d6a7b600d))
* k-neighbors-classifier await super.fit() ([01632f4](https://github.com/javascriptdata/scikit.js/commit/01632f45c98b82eb9e4a29d48c15973fd20ea22c))


### Features

* k-neighbors kd-tree algorithm ([59d40de](https://github.com/javascriptdata/scikit.js/commit/59d40de6eb9fb9d6105eb674e99a6251bdc2c0a5))
* kd-tree first draft ([354979a](https://github.com/javascriptdata/scikit.js/commit/354979a0bd5809a4eeab8ce877607cdd09f3660b))


### Performance Improvements

* k-neighbors kd-tree performance improvements ([158506c](https://github.com/javascriptdata/scikit.js/commit/158506c094202d4a645bb7b3eb7c45f2ef2c63e9))

# [1.1.0](https://github.com/javascriptdata/scikit.js/compare/v1.0.3...v1.1.0) (2021-12-31)


### Bug Fixes

* added fast-check dev dependency ([fe9e693](https://github.com/javascriptdata/scikit.js/commit/fe9e6939a0f0e70f55f1f2667fcbadacc2c1ca7f))
* change max length on commit message ([f4a8672](https://github.com/javascriptdata/scikit.js/commit/f4a86724f26776e9139dfb3bac8d1aa09a50a08d))
* commented out tests failing in test:browser ([0fe0fe1](https://github.com/javascriptdata/scikit.js/commit/0fe0fe1dba9ee2a42e7faf3e5a65d25d6a7b600d))
* k-neighbors-classifier await super.fit() ([01632f4](https://github.com/javascriptdata/scikit.js/commit/01632f45c98b82eb9e4a29d48c15973fd20ea22c))


### Features

* k-neighbors kd-tree algorithm ([59d40de](https://github.com/javascriptdata/scikit.js/commit/59d40de6eb9fb9d6105eb674e99a6251bdc2c0a5))
* kd-tree first draft ([354979a](https://github.com/javascriptdata/scikit.js/commit/354979a0bd5809a4eeab8ce877607cdd09f3660b))


### Performance Improvements

* k-neighbors kd-tree performance improvements ([158506c](https://github.com/javascriptdata/scikit.js/commit/158506c094202d4a645bb7b3eb7c45f2ef2c63e9))

## [1.0.3](https://github.com/javascriptdata/scikit.js/compare/v1.0.2...v1.0.3) (2021-12-31)


### Bug Fixes

* fixing any type to correct usage ([0084771](https://github.com/javascriptdata/scikit.js/commit/00847716d01d527fdd7ded7af3c718d6115e126a))

## [1.0.2](https://github.com/javascriptdata/scikit.js/compare/v1.0.1...v1.0.2) (2021-12-31)


### Bug Fixes

* fixing any type to correct usage ([3f5c288](https://github.com/javascriptdata/scikit.js/commit/3f5c288e32f3001a25fb15142f31165c8eceff4b))

## [1.0.1](https://github.com/javascriptdata/scikit.js/compare/v1.0.0...v1.0.1) (2021-12-31)


### Bug Fixes

* fixing any type to correct usage ([4496805](https://github.com/javascriptdata/scikit.js/commit/449680585706377167180b3e6c1c187958783b66))

# 1.0.0 (2021-12-31)


### Bug Fixes

* broken UMD browser script ([10f2e34](https://github.com/javascriptdata/scikit.js/commit/10f2e340faae2d3d9bd136839e1bcbddc149fa1b))
* fix lint ([28c876d](https://github.com/javascriptdata/scikit.js/commit/28c876d13eaa411bee7dfac6d8b6b3be428a1341))
* k-neighbors inverse distance weighting ([a162baa](https://github.com/javascriptdata/scikit.js/commit/a162baaedca307106b643b1b12a151517d351b1b))
* k-neighbors predict now checks n_features ([35efd93](https://github.com/javascriptdata/scikit.js/commit/35efd93bc450243c43b387498214b666bc15a841))


### Features

* add SVC ([a5fe596](https://github.com/javascriptdata/scikit.js/commit/a5fe596cd9d162ea9d4f2f48431b09e1b0865873))
* added cut 1 of voting classifier ([4045b81](https://github.com/javascriptdata/scikit.js/commit/4045b81aa1b91d51c0cade7f5e3cf52d1a307642))
* added tests and basic implementation of votingregressor ([d7011e7](https://github.com/javascriptdata/scikit.js/commit/d7011e74937284d283555992c180bb38706ebf6a))
* added voting classifier ([d7ab9c6](https://github.com/javascriptdata/scikit.js/commit/d7ab9c6e64efd5728f423b96dcac2112f8f5ae6f))
* broke out sgdlinear into sgdregressor and sgdclassifier ([81fbee8](https://github.com/javascriptdata/scikit.js/commit/81fbee82329e7f2f6d8865a9905e3044054b376d))
* changed imports ([390375c](https://github.com/javascriptdata/scikit.js/commit/390375c8a39d4e14af23542f6e18a67a16b9379f))
* finish ([825ebb7](https://github.com/javascriptdata/scikit.js/commit/825ebb7984e909ef212be27c925f85316278dee6))
* First pass at VotingRegressor ([ffb3393](https://github.com/javascriptdata/scikit.js/commit/ffb3393510bb3beacc56576e88569dd173013ee7))
* implemented kNeighborsRegression ([a1a7174](https://github.com/javascriptdata/scikit.js/commit/a1a7174ad261b2304012b4b285e8c9dad2118ab0))
* import libsvm ([f0f0cc8](https://github.com/javascriptdata/scikit.js/commit/f0f0cc899abe296ce9fc47a394674ac3e5bbcd8d))
* k-neighbors regressor ([94f6a69](https://github.com/javascriptdata/scikit.js/commit/94f6a6977ac46302b5431c962a8d07f06c4301f7))
* k-neighbors regressor ([225e167](https://github.com/javascriptdata/scikit.js/commit/225e16746af715380cbe67b090db057030feadc6))
* k-neighbors-classifier implemented ([d120257](https://github.com/javascriptdata/scikit.js/commit/d1202575f4562ea3a51a86e2df97675cbe4a17fe))
* k-neighbors-regressor ([050cec6](https://github.com/javascriptdata/scikit.js/commit/050cec6bf199e1b605b563d9a57d752e39c0ec8a)), closes [#111](https://github.com/javascriptdata/scikit.js/issues/111)
* k-neighbors-regressor ([cb0a8b0](https://github.com/javascriptdata/scikit.js/commit/cb0a8b07507e26791a722526def0e8e35b41c0ea))
* linear svr ([7a0534d](https://github.com/javascriptdata/scikit.js/commit/7a0534d17efeca68848353cdbdce9f61dd7b3aa5))
* simple first pass addition of linear-svc ([483117d](https://github.com/javascriptdata/scikit.js/commit/483117d818d3c6c003b25d18b2d6f16c2bc22554))
* train test split implementation ([97b89a5](https://github.com/javascriptdata/scikit.js/commit/97b89a50facbe150d0db9c83c2b499f824e6ead8))
* updated index to export linear-svc and updated docs ([b5c116b](https://github.com/javascriptdata/scikit.js/commit/b5c116bfb602a4524353662f026237b3bf06657d))

# 1.0.0 (2021-12-31)


### Bug Fixes

* broken UMD browser script ([10f2e34](https://github.com/javascriptdata/scikit.js/commit/10f2e340faae2d3d9bd136839e1bcbddc149fa1b))
* fix lint ([28c876d](https://github.com/javascriptdata/scikit.js/commit/28c876d13eaa411bee7dfac6d8b6b3be428a1341))
* k-neighbors inverse distance weighting ([a162baa](https://github.com/javascriptdata/scikit.js/commit/a162baaedca307106b643b1b12a151517d351b1b))
* k-neighbors predict now checks n_features ([35efd93](https://github.com/javascriptdata/scikit.js/commit/35efd93bc450243c43b387498214b666bc15a841))


### Features

* add SVC ([a5fe596](https://github.com/javascriptdata/scikit.js/commit/a5fe596cd9d162ea9d4f2f48431b09e1b0865873))
* added cut 1 of voting classifier ([4045b81](https://github.com/javascriptdata/scikit.js/commit/4045b81aa1b91d51c0cade7f5e3cf52d1a307642))
* added tests and basic implementation of votingregressor ([d7011e7](https://github.com/javascriptdata/scikit.js/commit/d7011e74937284d283555992c180bb38706ebf6a))
* added voting classifier ([d7ab9c6](https://github.com/javascriptdata/scikit.js/commit/d7ab9c6e64efd5728f423b96dcac2112f8f5ae6f))
* broke out sgdlinear into sgdregressor and sgdclassifier ([81fbee8](https://github.com/javascriptdata/scikit.js/commit/81fbee82329e7f2f6d8865a9905e3044054b376d))
* changed imports ([390375c](https://github.com/javascriptdata/scikit.js/commit/390375c8a39d4e14af23542f6e18a67a16b9379f))
* finish ([825ebb7](https://github.com/javascriptdata/scikit.js/commit/825ebb7984e909ef212be27c925f85316278dee6))
* First pass at VotingRegressor ([ffb3393](https://github.com/javascriptdata/scikit.js/commit/ffb3393510bb3beacc56576e88569dd173013ee7))
* implemented kNeighborsRegression ([a1a7174](https://github.com/javascriptdata/scikit.js/commit/a1a7174ad261b2304012b4b285e8c9dad2118ab0))
* import libsvm ([f0f0cc8](https://github.com/javascriptdata/scikit.js/commit/f0f0cc899abe296ce9fc47a394674ac3e5bbcd8d))
* k-neighbors regressor ([94f6a69](https://github.com/javascriptdata/scikit.js/commit/94f6a6977ac46302b5431c962a8d07f06c4301f7))
* k-neighbors regressor ([225e167](https://github.com/javascriptdata/scikit.js/commit/225e16746af715380cbe67b090db057030feadc6))
* k-neighbors-classifier implemented ([d120257](https://github.com/javascriptdata/scikit.js/commit/d1202575f4562ea3a51a86e2df97675cbe4a17fe))
* k-neighbors-regressor ([050cec6](https://github.com/javascriptdata/scikit.js/commit/050cec6bf199e1b605b563d9a57d752e39c0ec8a)), closes [#111](https://github.com/javascriptdata/scikit.js/issues/111)
* k-neighbors-regressor ([cb0a8b0](https://github.com/javascriptdata/scikit.js/commit/cb0a8b07507e26791a722526def0e8e35b41c0ea))
* linear svr ([7a0534d](https://github.com/javascriptdata/scikit.js/commit/7a0534d17efeca68848353cdbdce9f61dd7b3aa5))
* simple first pass addition of linear-svc ([483117d](https://github.com/javascriptdata/scikit.js/commit/483117d818d3c6c003b25d18b2d6f16c2bc22554))
* train test split implementation ([97b89a5](https://github.com/javascriptdata/scikit.js/commit/97b89a50facbe150d0db9c83c2b499f824e6ead8))
* updated index to export linear-svc and updated docs ([b5c116b](https://github.com/javascriptdata/scikit.js/commit/b5c116bfb602a4524353662f026237b3bf06657d))

# 1.0.0 (2021-12-31)


### Bug Fixes

* broken UMD browser script ([10f2e34](https://github.com/javascriptdata/scikit.js/commit/10f2e340faae2d3d9bd136839e1bcbddc149fa1b))
* fix lint ([28c876d](https://github.com/javascriptdata/scikit.js/commit/28c876d13eaa411bee7dfac6d8b6b3be428a1341))
* k-neighbors inverse distance weighting ([a162baa](https://github.com/javascriptdata/scikit.js/commit/a162baaedca307106b643b1b12a151517d351b1b))
* k-neighbors predict now checks n_features ([35efd93](https://github.com/javascriptdata/scikit.js/commit/35efd93bc450243c43b387498214b666bc15a841))


### Features

* add SVC ([a5fe596](https://github.com/javascriptdata/scikit.js/commit/a5fe596cd9d162ea9d4f2f48431b09e1b0865873))
* added cut 1 of voting classifier ([4045b81](https://github.com/javascriptdata/scikit.js/commit/4045b81aa1b91d51c0cade7f5e3cf52d1a307642))
* added tests and basic implementation of votingregressor ([d7011e7](https://github.com/javascriptdata/scikit.js/commit/d7011e74937284d283555992c180bb38706ebf6a))
* added voting classifier ([d7ab9c6](https://github.com/javascriptdata/scikit.js/commit/d7ab9c6e64efd5728f423b96dcac2112f8f5ae6f))
* broke out sgdlinear into sgdregressor and sgdclassifier ([81fbee8](https://github.com/javascriptdata/scikit.js/commit/81fbee82329e7f2f6d8865a9905e3044054b376d))
* changed imports ([390375c](https://github.com/javascriptdata/scikit.js/commit/390375c8a39d4e14af23542f6e18a67a16b9379f))
* finish ([825ebb7](https://github.com/javascriptdata/scikit.js/commit/825ebb7984e909ef212be27c925f85316278dee6))
* First pass at VotingRegressor ([ffb3393](https://github.com/javascriptdata/scikit.js/commit/ffb3393510bb3beacc56576e88569dd173013ee7))
* implemented kNeighborsRegression ([a1a7174](https://github.com/javascriptdata/scikit.js/commit/a1a7174ad261b2304012b4b285e8c9dad2118ab0))
* import libsvm ([f0f0cc8](https://github.com/javascriptdata/scikit.js/commit/f0f0cc899abe296ce9fc47a394674ac3e5bbcd8d))
* k-neighbors regressor ([94f6a69](https://github.com/javascriptdata/scikit.js/commit/94f6a6977ac46302b5431c962a8d07f06c4301f7))
* k-neighbors regressor ([225e167](https://github.com/javascriptdata/scikit.js/commit/225e16746af715380cbe67b090db057030feadc6))
* k-neighbors-classifier implemented ([d120257](https://github.com/javascriptdata/scikit.js/commit/d1202575f4562ea3a51a86e2df97675cbe4a17fe))
* k-neighbors-regressor ([050cec6](https://github.com/javascriptdata/scikit.js/commit/050cec6bf199e1b605b563d9a57d752e39c0ec8a)), closes [#111](https://github.com/javascriptdata/scikit.js/issues/111)
* k-neighbors-regressor ([cb0a8b0](https://github.com/javascriptdata/scikit.js/commit/cb0a8b07507e26791a722526def0e8e35b41c0ea))
* linear svr ([7a0534d](https://github.com/javascriptdata/scikit.js/commit/7a0534d17efeca68848353cdbdce9f61dd7b3aa5))
* simple first pass addition of linear-svc ([483117d](https://github.com/javascriptdata/scikit.js/commit/483117d818d3c6c003b25d18b2d6f16c2bc22554))
* train test split implementation ([97b89a5](https://github.com/javascriptdata/scikit.js/commit/97b89a50facbe150d0db9c83c2b499f824e6ead8))
* updated index to export linear-svc and updated docs ([b5c116b](https://github.com/javascriptdata/scikit.js/commit/b5c116bfb602a4524353662f026237b3bf06657d))

# 1.0.0 (2021-12-31)


### Bug Fixes

* broken UMD browser script ([10f2e34](https://github.com/javascriptdata/scikit.js/commit/10f2e340faae2d3d9bd136839e1bcbddc149fa1b))
* fix lint ([28c876d](https://github.com/javascriptdata/scikit.js/commit/28c876d13eaa411bee7dfac6d8b6b3be428a1341))
* k-neighbors inverse distance weighting ([a162baa](https://github.com/javascriptdata/scikit.js/commit/a162baaedca307106b643b1b12a151517d351b1b))
* k-neighbors predict now checks n_features ([35efd93](https://github.com/javascriptdata/scikit.js/commit/35efd93bc450243c43b387498214b666bc15a841))


### Features

* add SVC ([a5fe596](https://github.com/javascriptdata/scikit.js/commit/a5fe596cd9d162ea9d4f2f48431b09e1b0865873))
* added cut 1 of voting classifier ([4045b81](https://github.com/javascriptdata/scikit.js/commit/4045b81aa1b91d51c0cade7f5e3cf52d1a307642))
* added tests and basic implementation of votingregressor ([d7011e7](https://github.com/javascriptdata/scikit.js/commit/d7011e74937284d283555992c180bb38706ebf6a))
* added voting classifier ([d7ab9c6](https://github.com/javascriptdata/scikit.js/commit/d7ab9c6e64efd5728f423b96dcac2112f8f5ae6f))
* broke out sgdlinear into sgdregressor and sgdclassifier ([81fbee8](https://github.com/javascriptdata/scikit.js/commit/81fbee82329e7f2f6d8865a9905e3044054b376d))
* changed imports ([390375c](https://github.com/javascriptdata/scikit.js/commit/390375c8a39d4e14af23542f6e18a67a16b9379f))
* finish ([825ebb7](https://github.com/javascriptdata/scikit.js/commit/825ebb7984e909ef212be27c925f85316278dee6))
* First pass at VotingRegressor ([ffb3393](https://github.com/javascriptdata/scikit.js/commit/ffb3393510bb3beacc56576e88569dd173013ee7))
* implemented kNeighborsRegression ([a1a7174](https://github.com/javascriptdata/scikit.js/commit/a1a7174ad261b2304012b4b285e8c9dad2118ab0))
* import libsvm ([f0f0cc8](https://github.com/javascriptdata/scikit.js/commit/f0f0cc899abe296ce9fc47a394674ac3e5bbcd8d))
* k-neighbors regressor ([94f6a69](https://github.com/javascriptdata/scikit.js/commit/94f6a6977ac46302b5431c962a8d07f06c4301f7))
* k-neighbors regressor ([225e167](https://github.com/javascriptdata/scikit.js/commit/225e16746af715380cbe67b090db057030feadc6))
* k-neighbors-classifier implemented ([d120257](https://github.com/javascriptdata/scikit.js/commit/d1202575f4562ea3a51a86e2df97675cbe4a17fe))
* k-neighbors-regressor ([050cec6](https://github.com/javascriptdata/scikit.js/commit/050cec6bf199e1b605b563d9a57d752e39c0ec8a)), closes [#111](https://github.com/javascriptdata/scikit.js/issues/111)
* k-neighbors-regressor ([cb0a8b0](https://github.com/javascriptdata/scikit.js/commit/cb0a8b07507e26791a722526def0e8e35b41c0ea))
* linear svr ([7a0534d](https://github.com/javascriptdata/scikit.js/commit/7a0534d17efeca68848353cdbdce9f61dd7b3aa5))
* simple first pass addition of linear-svc ([483117d](https://github.com/javascriptdata/scikit.js/commit/483117d818d3c6c003b25d18b2d6f16c2bc22554))
* train test split implementation ([97b89a5](https://github.com/javascriptdata/scikit.js/commit/97b89a50facbe150d0db9c83c2b499f824e6ead8))
* updated index to export linear-svc and updated docs ([b5c116b](https://github.com/javascriptdata/scikit.js/commit/b5c116bfb602a4524353662f026237b3bf06657d))

# 1.0.0 (2021-12-31)


### Bug Fixes

* broken UMD browser script ([10f2e34](https://github.com/javascriptdata/scikit.js/commit/10f2e340faae2d3d9bd136839e1bcbddc149fa1b))
* fix lint ([28c876d](https://github.com/javascriptdata/scikit.js/commit/28c876d13eaa411bee7dfac6d8b6b3be428a1341))
* k-neighbors inverse distance weighting ([a162baa](https://github.com/javascriptdata/scikit.js/commit/a162baaedca307106b643b1b12a151517d351b1b))
* k-neighbors predict now checks n_features ([35efd93](https://github.com/javascriptdata/scikit.js/commit/35efd93bc450243c43b387498214b666bc15a841))


### Features

* add SVC ([a5fe596](https://github.com/javascriptdata/scikit.js/commit/a5fe596cd9d162ea9d4f2f48431b09e1b0865873))
* added cut 1 of voting classifier ([4045b81](https://github.com/javascriptdata/scikit.js/commit/4045b81aa1b91d51c0cade7f5e3cf52d1a307642))
* added tests and basic implementation of votingregressor ([d7011e7](https://github.com/javascriptdata/scikit.js/commit/d7011e74937284d283555992c180bb38706ebf6a))
* added voting classifier ([d7ab9c6](https://github.com/javascriptdata/scikit.js/commit/d7ab9c6e64efd5728f423b96dcac2112f8f5ae6f))
* broke out sgdlinear into sgdregressor and sgdclassifier ([81fbee8](https://github.com/javascriptdata/scikit.js/commit/81fbee82329e7f2f6d8865a9905e3044054b376d))
* changed imports ([390375c](https://github.com/javascriptdata/scikit.js/commit/390375c8a39d4e14af23542f6e18a67a16b9379f))
* finish ([825ebb7](https://github.com/javascriptdata/scikit.js/commit/825ebb7984e909ef212be27c925f85316278dee6))
* First pass at VotingRegressor ([ffb3393](https://github.com/javascriptdata/scikit.js/commit/ffb3393510bb3beacc56576e88569dd173013ee7))
* implemented kNeighborsRegression ([a1a7174](https://github.com/javascriptdata/scikit.js/commit/a1a7174ad261b2304012b4b285e8c9dad2118ab0))
* import libsvm ([f0f0cc8](https://github.com/javascriptdata/scikit.js/commit/f0f0cc899abe296ce9fc47a394674ac3e5bbcd8d))
* k-neighbors regressor ([94f6a69](https://github.com/javascriptdata/scikit.js/commit/94f6a6977ac46302b5431c962a8d07f06c4301f7))
* k-neighbors regressor ([225e167](https://github.com/javascriptdata/scikit.js/commit/225e16746af715380cbe67b090db057030feadc6))
* k-neighbors-classifier implemented ([d120257](https://github.com/javascriptdata/scikit.js/commit/d1202575f4562ea3a51a86e2df97675cbe4a17fe))
* k-neighbors-regressor ([050cec6](https://github.com/javascriptdata/scikit.js/commit/050cec6bf199e1b605b563d9a57d752e39c0ec8a)), closes [#111](https://github.com/javascriptdata/scikit.js/issues/111)
* k-neighbors-regressor ([cb0a8b0](https://github.com/javascriptdata/scikit.js/commit/cb0a8b07507e26791a722526def0e8e35b41c0ea))
* linear svr ([7a0534d](https://github.com/javascriptdata/scikit.js/commit/7a0534d17efeca68848353cdbdce9f61dd7b3aa5))
* simple first pass addition of linear-svc ([483117d](https://github.com/javascriptdata/scikit.js/commit/483117d818d3c6c003b25d18b2d6f16c2bc22554))
* train test split implementation ([97b89a5](https://github.com/javascriptdata/scikit.js/commit/97b89a50facbe150d0db9c83c2b499f824e6ead8))
* updated index to export linear-svc and updated docs ([b5c116b](https://github.com/javascriptdata/scikit.js/commit/b5c116bfb602a4524353662f026237b3bf06657d))

# 1.0.0 (2021-12-31)


### Bug Fixes

* broken UMD browser script ([10f2e34](https://github.com/javascriptdata/scikit.js/commit/10f2e340faae2d3d9bd136839e1bcbddc149fa1b))
* fix lint ([28c876d](https://github.com/javascriptdata/scikit.js/commit/28c876d13eaa411bee7dfac6d8b6b3be428a1341))
* k-neighbors inverse distance weighting ([a162baa](https://github.com/javascriptdata/scikit.js/commit/a162baaedca307106b643b1b12a151517d351b1b))
* k-neighbors predict now checks n_features ([35efd93](https://github.com/javascriptdata/scikit.js/commit/35efd93bc450243c43b387498214b666bc15a841))


### Features

* add SVC ([a5fe596](https://github.com/javascriptdata/scikit.js/commit/a5fe596cd9d162ea9d4f2f48431b09e1b0865873))
* added cut 1 of voting classifier ([4045b81](https://github.com/javascriptdata/scikit.js/commit/4045b81aa1b91d51c0cade7f5e3cf52d1a307642))
* added tests and basic implementation of votingregressor ([d7011e7](https://github.com/javascriptdata/scikit.js/commit/d7011e74937284d283555992c180bb38706ebf6a))
* added voting classifier ([d7ab9c6](https://github.com/javascriptdata/scikit.js/commit/d7ab9c6e64efd5728f423b96dcac2112f8f5ae6f))
* broke out sgdlinear into sgdregressor and sgdclassifier ([81fbee8](https://github.com/javascriptdata/scikit.js/commit/81fbee82329e7f2f6d8865a9905e3044054b376d))
* changed imports ([390375c](https://github.com/javascriptdata/scikit.js/commit/390375c8a39d4e14af23542f6e18a67a16b9379f))
* finish ([825ebb7](https://github.com/javascriptdata/scikit.js/commit/825ebb7984e909ef212be27c925f85316278dee6))
* First pass at VotingRegressor ([ffb3393](https://github.com/javascriptdata/scikit.js/commit/ffb3393510bb3beacc56576e88569dd173013ee7))
* implemented kNeighborsRegression ([a1a7174](https://github.com/javascriptdata/scikit.js/commit/a1a7174ad261b2304012b4b285e8c9dad2118ab0))
* import libsvm ([f0f0cc8](https://github.com/javascriptdata/scikit.js/commit/f0f0cc899abe296ce9fc47a394674ac3e5bbcd8d))
* k-neighbors regressor ([94f6a69](https://github.com/javascriptdata/scikit.js/commit/94f6a6977ac46302b5431c962a8d07f06c4301f7))
* k-neighbors regressor ([225e167](https://github.com/javascriptdata/scikit.js/commit/225e16746af715380cbe67b090db057030feadc6))
* k-neighbors-classifier implemented ([d120257](https://github.com/javascriptdata/scikit.js/commit/d1202575f4562ea3a51a86e2df97675cbe4a17fe))
* k-neighbors-regressor ([050cec6](https://github.com/javascriptdata/scikit.js/commit/050cec6bf199e1b605b563d9a57d752e39c0ec8a)), closes [#111](https://github.com/javascriptdata/scikit.js/issues/111)
* k-neighbors-regressor ([cb0a8b0](https://github.com/javascriptdata/scikit.js/commit/cb0a8b07507e26791a722526def0e8e35b41c0ea))
* linear svr ([7a0534d](https://github.com/javascriptdata/scikit.js/commit/7a0534d17efeca68848353cdbdce9f61dd7b3aa5))
* simple first pass addition of linear-svc ([483117d](https://github.com/javascriptdata/scikit.js/commit/483117d818d3c6c003b25d18b2d6f16c2bc22554))
* train test split implementation ([97b89a5](https://github.com/javascriptdata/scikit.js/commit/97b89a50facbe150d0db9c83c2b499f824e6ead8))
* updated index to export linear-svc and updated docs ([b5c116b](https://github.com/javascriptdata/scikit.js/commit/b5c116bfb602a4524353662f026237b3bf06657d))
