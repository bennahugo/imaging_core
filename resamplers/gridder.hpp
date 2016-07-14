#pragma once

#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>

#include "uv_data.hpp"

namespace imaging_core{
  namespace resamplers{
    using namespace boost::python;
    using namespace std;
    void grid(uv_data input_uv){       
      cout << input_uv.m_uvw[0] << input_uv.m_uvw[1] << input_uv.m_uvw[2] << endl;
    }
  }
}



