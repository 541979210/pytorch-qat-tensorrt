# -*- coding: utf-8 -*-
"""
@Author:
@license: MIT-license

This is a demo python script

Overview:

- fun print hello world
"""
import sys
import os
import time
import tensorrt as trt
import cv2
import numpy as np
'''
    engine: 推理用到的模型
    builder: 用来构建engine
    config:
    parser: 用来解析onnx文件
'''
sys.path.append(r'D:\code\whl\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8\TensorRT-8.6.1.6\samples\python')#common 文件的位置
import common
TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path):
    '''
    Attempts to load a serialized engine if available,
    otherwise build a new TensorRT engine as save it
    '''

    def build_engine():
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)
        runtime = trt.Runtime(TRT_LOGGER)

        config.max_workspace_size = 1 << 30  # 256MB 最大内存占用
        builder.max_batch_size = 2  # 推理的时候要保证batch_size<=max_batch_size

        # parse model file
        if not os.path.exists(onnx_file_path):
            print(f'onnx file {onnx_file_path} not found,please run torch_2_onnx.py first to generate it')
            exit(0)
        print(f'Loading ONNX file from path {onnx_file_path}...')
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR:Failed to parse the ONNX file')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Static input setting
        network.get_input(0).shape = [1, 3, 224, 224]
        # Dynamic input setting 动态输入在builder里面设置
        # profile = builder.create_optimization_profile()
        # profile.set_shape('input',(1,3,224,224),(1,3,512,512),(1,3,1024,1024))#最小的尺寸,常用的尺寸,最大的尺寸,推理时候输入需要在这个范围内
        # config.add_optimization_profile(profile)

        config.set_flag(trt.BuilderFlag.INT8)
        print('Completed parsing the ONNX file')
        print(f'Building an engine from file {onnx_file_path}; this may take a while...')
        # plan = builder.build_serialized_network(network,config)
        # engine = runtime.deserialize_cuda_engine(plan)
        engine = builder.build_engine(network, config)
        print('Completed creating Engine')
        with open(engine_file_path, 'wb') as f:
            # f.write(plan)
            f.write(engine.serialize())
        return engine

    if os.path.exists(engine_file_path):
        print(f'Reading engine from file {engine_file_path}')
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main(out_dir, model_name):
    # create trt engine for onnx-based portrait model
    onnx_file_path = out_dir + '/' + model_name + ".onnx"
    trt_file_path = out_dir + '/' + model_name + ".trt"

    # get trt model
    engine = get_engine(onnx_file_path,trt_file_path)
    # do inference with trt engine
    context = engine.create_execution_context()
    inputs,outputs,bindings,stream = common.allocate_buffers(engine)
    print(f'Running inference on fake image...')
    tmpImg = np.ascontiguousarray(np.random.rand(1,3,224,224),dtype=np.float16)
    inputs[0].host = np.ascontiguousarray(tmpImg) #************************
    trt_outputs = common.do_inference_v2(context,bindings=bindings,inputs=inputs,outputs=outputs,stream=stream)[0]


if __name__=='__main__':
    main('./tmp', 'resnet50')
