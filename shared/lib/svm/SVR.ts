import { dfd, tf } from 'globals';
import { Scikit1D, Scikit2D } from '../index';
import { isScikitLike2D } from '../typesUtils';
//@ts-ignore
import { SVM, SVMParam, KERNEL_TYPE, ISVMParam } from 'libsvm-wasm';

export interface SVRParams {
    kernel?: 'LINEAR' | 'POLY' | 'RBF' | 'SIGMOID' | 'PRECOMPUTED';
    degree?: number;
    gamma?: number | 'auto' | 'scale';
    coef0?: number;
    tol?: number;
    C?: number;
    epsilon?: number;
    shrinking?: boolean;
    cacheSize?: number;
    maxIter?: number;
}

export class SVR {
    private svm?: SVM;
    private svmParam: SVMParam;
    private gammaMode: string = 'scale';

    constructor({
        kernel = 'RBF',
        degree = 3,
        gamma = 'scale',
        coef0 = 0,
        tol = 1e-3,
        C = 1,
        epsilon = 0.1,
        shrinking = true,
        cacheSize = 200,
        maxIter = -1
    }: SVRParams = {}) {
        const inernalSVMParam: ISVMParam = {
            kernel_type: KERNEL_TYPE[kernel],
            degree,
            coef0,
            C,
            p: epsilon,
            shrinking: shrinking ? 1 : 0,
            cache_size: cacheSize,
        }
        if (gamma === 'auto') {
            inernalSVMParam.gamma = -1;
            this.gammaMode = gamma;
        } else if (gamma === 'scale') {
            inernalSVMParam.gamma = -2;
            this.gammaMode = gamma;
        } else {
            inernalSVMParam.gamma = gamma;
        }

        this.svmParam = new SVMParam(inernalSVMParam, tol);
    }

    private async processInput(X: Scikit2D | Scikit1D): Promise<any> {
        let rawData: any = [];
        if (X instanceof tf.Tensor) {
            rawData = await X.array();
        } else if (X instanceof dfd.DataFrame) {
            rawData = X.values;
        } else {
            rawData = X;
        }
        return rawData;
    }

    async fit(X: Scikit2D, y: Scikit1D): Promise<SVR> {
        let nSample: number, nFeature: number;
        if (isScikitLike2D(X)) {
            nSample = X.length;
            nFeature = X[0].length;
        } else {
            nSample = X.shape[0];
            nFeature = X.shape[1];
        }
        
        // TODO: should apply variance of X
        if (this.gammaMode === 'scale') {
            this.svmParam.param.gamma = 1 / nFeature;
        } else if (this.gammaMode === 'auto') {
            this.svmParam.param.gamma = 1 / nFeature;
        }

        const [processX, processY] = await Promise.all([this.processInput(X), this.processInput(y)]);
        this.svm = new SVM(this.svmParam);
        await this.svm.feedSamples(processX, processY);
        await this.svm.train();
        return this;
    }

    async predict(X: Scikit2D): Promise<Scikit1D> {
        const processX = await Promise.all([this.processInput(X)]);
        if (this.svm) {
            const results = []
            for (let i = 0; i < processX.length; i++) {
                const result = this.svm.predict(processX[i]);
                results.push(result);
            }
            return Promise.all(results);
        } else {
            throw new Error('SVM not trained');
        }
    }
}

