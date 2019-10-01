#ifndef _DPU_RESIZE_H_
#define _DPU_RESIZE_H_

#include <fstream>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <vector>

using namespace std;

struct _param {
  uint16_t start_x;
  uint16_t end_x;
  uint16_t start_y;
  uint16_t end_y;
  uint16_t frac_x[2];
  uint16_t frac_y[2];
};
struct _config {
  uint32_t scale_w;
  uint32_t scale_h;
  uint16_t src_w;
  uint16_t src_h;
  uint16_t src_c;
  uint16_t dst_w;
  uint16_t dst_h;
  uint16_t dst_c;
  uint16_t inter_mode;
};

class dpu_resize {
private:
  int CQBIT;
  int IM_LINEAR;
  int IM_NEAREST;
  int IM_MAX;

  struct _config cfg;
  struct _param **p_matrix;
  uint8_t *img_src;
  uint8_t *img_dst;

  void param_gen();

public:
  dpu_resize(uint8_t *ext_img_src, uint8_t *exi_img_dst,
             struct _config ext_cfg);
  void calc();
  void dump_config();
  void dump_img_src();
  void dump_img_dst();
  void dump_param();
};

#endif
