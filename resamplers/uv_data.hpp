#pragma once
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <numpy/arrayobject.h>
namespace imaging_core{
  namespace resamplers{
    using namespace std;
    using namespace boost::python;
    struct uv_data
    {
      double * m_uvw;
      size_t m_num_rows;
      
      uv_data(numeric::array & uvw){
	size_t rows = extract<size_t>(uvw.attr("shape")[0]);
	size_t l = len(uvw.attr("shape"));
	
	double u = extract<double>(uvw[make_tuple(0,0)]);
	double v = extract<double>(uvw[make_tuple(0,1)]);
	double w = extract<double>(uvw[make_tuple(0,2)]);
	bool cont = uvw.attr("dtype");
	cout << cont << endl;
	cout << l << endl;
	cout << rows << endl;
	cout << u << " " << v << " " << w << endl;
	
	m_uvw = reinterpret_cast<double*>(reinterpret_cast<PyArrayObject*>(uvw.ptr())->data);
	m_num_rows = rows;
      }
    };
  }
}