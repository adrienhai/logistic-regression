#include <iostream>
#include <cmath>
#include "armadillo.hpp"

arma::vec logisticGradient (arma::vec y, arma::mat x, arma::vec w){
  double N=x.n_rows;
  arma::vec out = arma::vec(x.n_cols,arma::fill::zeros);
  for (size_t i = 0; i < N; i++) {
    out = out + y(i)/(1+exp(y(i)*dot(x.row(i),w)))*x.row(i).t();
  }
  return -out/N;
}

double f(arma::vec x, arma::vec w){
  return dot(x,w);
}

int main(int argc, char const *argv[]) {
  arma::vec dataY;
  arma::mat dataX,dataXtest;
  dataY.load("dataY.dat");
  dataX.load("dataX.dat");
  dataXtest.load("dataXtest.dat");

  arma::vec w = arma::vec(dataX.n_cols,arma::fill::zeros);
  double eps=pow(10,-7);
  arma::vec grad;
  do {
    grad=logisticGradient(dataY,dataX,w);
    w = w - 0.7*grad;
  } while(arma::norm(grad)>=eps);
  //std::cout << f(dataXtest.row(1).t(),w) << '\n';
  // Writes output file
	std::ofstream write_output("LogReg.dat") ;
	assert(write_output.is_open());
	for(int i = 0; i<dataXtest.n_rows; i++)
	{
    double f_x=f(dataXtest.row(i).t(),w);
    if (f_x>0)
      write_output << 1 << "\n" ;
    else
      write_output << -1 << "\n";
	}
	write_output.close();

  return 0;

}
