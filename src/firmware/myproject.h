#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"
// hls-fpga-machine-learning insert weights
#include "weights/e5.h"
#include "weights/e6.h"
#include "weights/s11.h"
#include "weights/b11.h"
#include "weights/w13.h"
#include "weights/b13.h"
#include "weights/w16.h"
#include "weights/b16.h"
#include "weights/w19.h"
#include "weights/b19.h"
#include "weights/w26.h"
#include "weights/b26.h"
#include "weights/s31.h"
#include "weights/b31.h"
#include "weights/w76.h"
#include "weights/b76.h"
#include "weights/w77.h"
#include "weights/b77.h"
#include "weights/s42.h"
#include "weights/b42.h"
#include "weights/w44.h"
#include "weights/b44.h"
#include "weights/w47.h"
#include "weights/b47.h"
#include "weights/w50.h"
#include "weights/b50.h"
#include "weights/w57.h"
#include "weights/b57.h"
#include "weights/s62.h"
#include "weights/b62.h"
#include "weights/w78.h"
#include "weights/b78.h"
#include "weights/w79.h"
#include "weights/b79.h"
#include "weights/s73.h"
#include "weights/b73.h"
#include "weights/w80.h"
#include "weights/b80.h"

// This file defines the interface to the kernel

// currently this is fixed
using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(sycl::ext::altera::experimental::ready_latency<0>));

// Need to declare the input and output pipes

// hls-fpga-machine-learning insert inputs
class TokenInputsPipeID;
using TokenInputsPipe = sycl::ext::altera::experimental::pipe<TokenInputsPipeID, token_inputs_t, 48, PipeProps>;
class PosInputsPipeID;
using PosInputsPipe = sycl::ext::altera::experimental::pipe<PosInputsPipeID, pos_inputs_t, 48, PipeProps>;
// hls-fpga-machine-learning insert outputs
class Layer80OutPipeID;
using token_arr_t = typename nnet::array<typename token_inputs_t::value_type, 1>;
using Layer80OutPipe = sycl::ext::altera::experimental::pipe<Layer80OutPipeID, token_arr_t, 48, PipeProps>;

class MyprojectID;

struct Myproject {

    // kernel property method to config invocation interface
    auto get(sycl::ext::oneapi::experimental::properties_tag) {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::altera::experimental::streaming_interface<>,
                                                           /*sycl::ext::altera::experimental::pipelined<>*/};
    }

    SYCL_EXTERNAL void operator()() const;
};

#endif
