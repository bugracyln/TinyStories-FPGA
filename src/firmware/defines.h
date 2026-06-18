#ifndef DEFINES_H_
#define DEFINES_H_

#include <sycl/ext/altera/ac_types/ac_fixed.hpp>
#include <sycl/ext/altera/ac_types/ac_int.hpp>
#include <sycl/ext/altera/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ac_fixed<10,10,false,AC_RND,AC_WRAP>, 32*1> token_inputs_t;
typedef nnet::array<ac_fixed<8,8,false,AC_RND,AC_WRAP>, 32*1> pos_inputs_t;
typedef nnet::array<ac_fixed<9,3,true,AC_RND,AC_WRAP>, 32*1> embedding_8_t;
typedef nnet::array<ac_fixed<9,3,true>, 32768*1> e5_t;
typedef nnet::array<ac_fixed<9,2,true,AC_RND,AC_WRAP>, 32*1> embedding_9_t;
typedef nnet::array<ac_fixed<9,2,true>, 1536*1> e6_t;
typedef nnet::array<ac_fixed<10,3,true>, 32*1> q_add_16_t;
typedef nnet::array<ac_fixed<7,3,true>, 32*1> q_bn_1_0_t;
typedef nnet::array<ac_fixed<8,3,false>, 32*1> s11_t;
typedef nnet::array<ac_fixed<9,-1,true>, 32*1> b11_t;
typedef ac_fixed<13,5,true> q_mha_0_query_accum_t;
typedef nnet::array<ac_fixed<4,1,true,AC_RND,AC_SAT>, 32*1> q_mha_0_query_t;
typedef nnet::array<ac_fixed<6,1,true>, 1024*1> w13_t;
typedef nnet::array<ac_fixed<8,1,true>, 1536*1> b13_t;
typedef ac_fixed<15,5,true> q_mha_0_key_accum_t;
typedef nnet::array<ac_fixed<6,3,true,AC_RND,AC_SAT>, 32*1> q_mha_0_key_t;
typedef nnet::array<ac_fixed<6,1,true>, 1024*1> w16_t;
typedef nnet::array<ac_fixed<2,32,false>, 1536*1> b16_t;
typedef ac_fixed<17,6,true> q_mha_0_value_accum_t;
typedef nnet::array<ac_fixed<6,2,true,AC_RND,AC_SAT>, 32*1> q_mha_0_value_t;
typedef nnet::array<ac_fixed<8,1,true>, 1024*1> w19_t;
typedef nnet::array<ac_fixed<9,-1,true>, 1536*1> b19_t;
typedef ac_fixed<15,9,true> q_mha_0_q_mha_0_QK_accum_t;
typedef nnet::array<ac_fixed<15,9,true>, 48*1> q_mha_0_q_mha_0_QK_t;
typedef ac_fixed<5,2,false,AC_RND_CONV,AC_SAT> q_mha_0_q_softmax_exp_table_t;
typedef ac_fixed<5,2,false,AC_RND_CONV,AC_SAT> q_mha_0_q_softmax_inv_table_t;
typedef nnet::array<ac_fixed<5,2,false,AC_RND_CONV,AC_SAT>, 32*1> q_mha_0_q_softmax_exp_table_arr_t;
typedef nnet::array<ac_fixed<5,2,false,AC_RND_CONV,AC_SAT>, 32*1> q_mha_0_q_softmax_inv_table_arr_t;
typedef ac_fixed<6,3,true,AC_RND,AC_SAT> q_mha_0_q_softmax_inv_inp_t;
typedef ac_fixed<6,3,true,AC_RND,AC_SAT> q_mha_0_q_softmax_inp_norm_t;
typedef ac_fixed<24,9,true> q_mha_0_q_softmax_accum_t;
typedef nnet::array<ac_fixed<4,-1,false,AC_RND,AC_SAT>, 48*1> q_mha_0_q_softmax_t;
typedef ac_fixed<18,8,true> q_mha_0_q_softmax_table_t;
typedef ac_fixed<16,7,true> q_mha_0_q_mha_0_aV_accum_t;
typedef nnet::array<ac_fixed<9,3,true,AC_RND,AC_SAT>, 32*1> q_mha_0_q_mha_0_aV_t;
typedef ac_fixed<20,7,true> q_mha_0_attention_output_accum_t;
typedef nnet::array<ac_fixed<15,5,true,AC_RND,AC_WRAP>, 32*1> q_mha_0_attention_output_t;
typedef nnet::array<ac_fixed<8,1,true>, 1024*1> w26_t;
typedef nnet::array<ac_fixed<3,2,true>, 1536*1> b26_t;
typedef nnet::array<ac_fixed<15,5,true>, 32*1> q_add_17_t;
typedef nnet::array<ac_fixed<8,4,true,AC_RND,AC_WRAP>, 32*1> q_bn_2_0_t;
typedef nnet::array<ac_fixed<8,2,false>, 32*1> s31_t;
typedef nnet::array<ac_fixed<9,2,true>, 32*1> b31_t;
typedef ac_fixed<17,7,true> q_dense_12_accum_t;
typedef nnet::array<ac_fixed<12,7,true>, 64*1> q_dense_12_t;
typedef nnet::array<ac_fixed<7,1,true>, 2048*1> w76_t;
typedef nnet::array<ac_fixed<9,2,true>, 64*1> b76_t;
typedef nnet::array<ac_fixed<8,4,true,AC_RND,AC_WRAP>, 64*1> q_dense_12_relu_t;
typedef ac_fixed<18,8,true> q_dense_12_relu_table_t;
typedef ac_fixed<19,8,true> q_dense_13_accum_t;
typedef nnet::array<ac_fixed<18,8,true>, 32*1> q_dense_13_t;
typedef nnet::array<ac_fixed<9,2,true>, 2048*1> w77_t;
typedef nnet::array<ac_fixed<9,3,true>, 32*1> b77_t;
typedef nnet::array<ac_fixed<14,5,true,AC_RND,AC_WRAP>, 32*1> q_dense_13_relu_t;
typedef ac_fixed<18,8,true> q_dense_13_relu_table_t;
typedef nnet::array<ac_fixed<14,5,true>, 32*1> q_add_18_t;
typedef nnet::array<ac_fixed<6,1,true>, 32*1> q_bn_1_1_t;
typedef nnet::array<ac_fixed<8,0,false>, 32*1> s42_t;
typedef nnet::array<ac_fixed<9,2,true>, 32*1> b42_t;
typedef ac_fixed<10,4,true> q_mha_1_query_accum_t;
typedef nnet::array<ac_fixed<2,0,true,AC_RND,AC_SAT>, 32*1> q_mha_1_query_t;
typedef nnet::array<ac_fixed<4,1,true>, 1024*1> w44_t;
typedef nnet::array<ac_fixed<6,0,true>, 1536*1> b44_t;
typedef ac_fixed<12,5,true> q_mha_1_key_accum_t;
typedef nnet::array<ac_fixed<4,1,true,AC_RND,AC_SAT>, 32*1> q_mha_1_key_t;
typedef nnet::array<ac_fixed<4,1,true>, 1024*1> w47_t;
typedef nnet::array<ac_fixed<2,32,false>, 1536*1> b47_t;
typedef ac_fixed<16,4,true> q_mha_1_value_accum_t;
typedef nnet::array<ac_fixed<5,1,true,AC_RND,AC_SAT>, 32*1> q_mha_1_value_t;
typedef nnet::array<ac_fixed<8,1,true>, 1024*1> w50_t;
typedef nnet::array<ac_fixed<8,-1,true>, 1536*1> b50_t;
typedef ac_fixed<11,6,true> q_mha_1_q_mha_1_QK_accum_t;
typedef nnet::array<ac_fixed<11,6,true>, 48*1> q_mha_1_q_mha_1_QK_t;
typedef ac_fixed<5,2,false,AC_RND_CONV,AC_SAT> q_mha_1_q_softmax_1_exp_table_t;
typedef ac_fixed<5,2,false,AC_RND_CONV,AC_SAT> q_mha_1_q_softmax_1_inv_table_t;
typedef nnet::array<ac_fixed<5,2,false,AC_RND_CONV,AC_SAT>, 32*1> q_mha_1_q_softmax_1_exp_table_arr_t;
typedef nnet::array<ac_fixed<5,2,false,AC_RND_CONV,AC_SAT>, 32*1> q_mha_1_q_softmax_1_inv_table_arr_t;
typedef ac_fixed<6,3,true,AC_RND,AC_SAT> q_mha_1_q_softmax_1_inv_inp_t;
typedef ac_fixed<6,3,true,AC_RND,AC_SAT> q_mha_1_q_softmax_1_inp_norm_t;
typedef ac_fixed<24,9,true> q_mha_1_q_softmax_1_accum_t;
typedef nnet::array<ac_fixed<4,-1,false,AC_RND,AC_SAT>, 48*1> q_mha_1_q_softmax_1_t;
typedef ac_fixed<18,8,true> q_mha_1_q_softmax_1_table_t;
typedef ac_fixed<15,6,true> q_mha_1_q_mha_1_aV_accum_t;
typedef nnet::array<ac_fixed<7,3,true,AC_RND,AC_SAT>, 32*1> q_mha_1_q_mha_1_aV_t;
typedef ac_fixed<16,6,true> q_mha_1_attention_output_accum_t;
typedef nnet::array<ac_fixed<13,4,true,AC_RND,AC_WRAP>, 32*1> q_mha_1_attention_output_t;
typedef nnet::array<ac_fixed<7,1,true>, 1024*1> w57_t;
typedef nnet::array<ac_fixed<3,3,true>, 1536*1> b57_t;
typedef nnet::array<ac_fixed<13,4,true>, 32*1> q_add_19_t;
typedef nnet::array<ac_fixed<8,4,true,AC_RND,AC_WRAP>, 32*1> q_bn_2_1_t;
typedef nnet::array<ac_fixed<8,2,false>, 32*1> s62_t;
typedef nnet::array<ac_fixed<9,2,true>, 32*1> b62_t;
typedef ac_fixed<18,7,true> q_dense_14_accum_t;
typedef nnet::array<ac_fixed<12,7,true>, 64*1> q_dense_14_t;
typedef nnet::array<ac_fixed<8,1,true>, 2048*1> w78_t;
typedef nnet::array<ac_fixed<8,1,true>, 64*1> b78_t;
typedef nnet::array<ac_fixed<8,4,true,AC_RND,AC_WRAP>, 64*1> q_dense_14_relu_t;
typedef ac_fixed<18,8,true> q_dense_14_relu_table_t;
typedef ac_fixed<19,9,true> q_dense_15_accum_t;
typedef nnet::array<ac_fixed<18,9,true>, 32*1> q_dense_15_t;
typedef nnet::array<ac_fixed<9,3,true>, 2048*1> w79_t;
typedef nnet::array<ac_fixed<9,4,true>, 32*1> b79_t;
typedef nnet::array<ac_fixed<13,5,true,AC_RND,AC_WRAP>, 32*1> q_dense_15_relu_t;
typedef ac_fixed<18,8,true> q_dense_15_relu_table_t;
typedef nnet::array<ac_fixed<11,5,true,AC_RND,AC_WRAP>, 32*1> q_add_20_t;
typedef nnet::array<ac_fixed<6,5,true,AC_RND,AC_WRAP>, 32*1> q_bn_final_t;
typedef nnet::array<ac_fixed<8,2,false>, 32*1> s73_t;
typedef nnet::array<ac_fixed<9,4,true>, 32*1> b73_t;
typedef ac_fixed<16,10,true> lm_head_accum_t;
typedef nnet::array<ac_fixed<16,10,true>, 1024*1> result_t;
typedef nnet::array<ac_fixed<7,2,true>, 32768*1> w80_t;
typedef nnet::array<ac_fixed<7,4,true>, 1024*1> b80_t;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif
