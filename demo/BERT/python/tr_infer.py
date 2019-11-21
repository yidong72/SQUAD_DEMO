#!/usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import ctypes
import numpy as np
import tokenization
import tensorrt as trt
import data_processing as dp
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
class Model(object):

    def __init__(self):
        vocab_file = '../vocab.txt'
        bert_engine = '../bert_base_384.engine'
        paragraph_text =  "today is a good day to play."
        question_text = "today is what?"

        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        # The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
        self.max_query_length = 64
        # When splitting up a long document into chunks, how much stride to take between chunks.
        self.doc_stride = 128
        # The maximum total input sequence length after WordPiece tokenization.
        # Sequences longer than this will be truncated, and sequences shorter
        self.max_seq_length = 384
        # Extract tokecs from the paragraph
        # Import necessary plugins for BERT TensorRT
        ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libcommon.so", mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libbert_plugins.so", mode=ctypes.RTLD_GLOBAL)
        f = open(bert_engine, 'rb')
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        f.close()

        input_shape = (1, self.max_seq_length)
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize

        # Allocate device memory for inputs.
        self.d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()

        # Specify input shapes. These must be within the min/max bounds of the active profile (0th profile in this case)
        # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
        for binding in range(3):
            self.context.set_binding_shape(binding, input_shape)
        assert self.context.all_binding_shapes_specified

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        self.h_output = cuda.pagelocked_empty(tuple(self.context.get_binding_shape(3)), dtype=np.float32)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

    def question_features(self, paragraph_text, question):
        doc_tokens = dp.convert_doc_tokens(paragraph_text)
        # Extract features from the paragraph and question
        return dp.convert_examples_to_features(doc_tokens, question, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length), doc_tokens

    # The first context created will use the 0th profile. A new context must be created
    # for each additional profile needed. Here, we only use batch size 1, thus we only need the first profile.

    def inference(self, paragraph_text, question):
        features, doc_tokens = self.question_features(paragraph_text, question)
        print("\nRunning Inference...")
        eval_start_time = time.time()

        # Copy inputs
        cuda.memcpy_htod_async(self.d_inputs[0], features["input_ids"], self.stream)
        cuda.memcpy_htod_async(self.d_inputs[1], features["segment_ids"], self.stream)
        cuda.memcpy_htod_async(self.d_inputs[2], features["input_mask"], self.stream)
        # Run inference
        self.context.execute_async_v2(bindings=[int(d_inp) for d_inp in self.d_inputs] + [int(self.d_output)], stream_handle=self.stream.handle)
        # Transfer predictions back from GPU
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        # Synchronize the stream
        self.stream.synchronize()

        eval_time_elapsed = time.time() - eval_start_time

        print("------------------------")
        print("Running inference in {:.3f} Sentences/Sec".format(1.0/eval_time_elapsed))
        print("------------------------")

        for index, batch in enumerate(self.h_output):
            # Data Post-processing
            start_logits = batch[:, 0]
            end_logits = batch[:, 1]

            # Total number of n-best predictions to generate in the nbest_predictions.json output file
            n_best_size = 20

            # The maximum length of an answer that can be generated. This is needed
            # because the start and end predictions are not conditioned on one another
            max_answer_length = 30

            prediction, nbest_json, scores_diff_json = dp.get_predictions(doc_tokens, features,
                    start_logits, end_logits, n_best_size, max_answer_length)
            output = {}
            output['result']=prediction
            output['p'] = nbest_json[0]['probability'] * 100.0
            print("Processing output {:} in batch".format(index))
            print("Answer: '{}'".format(prediction))
            print("With probability: {:.3f}".format(nbest_json[0]['probability'] * 100.0))
            return output

if __name__=="__main__":
    vocab_file = '../vocab.txt'
    bert_engine = '../bert_base_384.engine'
    paragraph_text =  "today is a good day to play."
    question_text = "today is what?"
    m = Model()
    m.inference(paragraph_text, question_text)
    paragraph_text =  "she is a liar."
    question_text = "who is she?"
    m.inference(paragraph_text, question_text)
