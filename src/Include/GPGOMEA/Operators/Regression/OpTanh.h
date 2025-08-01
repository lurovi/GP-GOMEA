/*
 


 */

/* 
 * File:   OpTanh.h
 * Author: rovito
 *
 * Created on July 30, 2025, 12:37 PM
 */

#ifndef OPTANH_H
#define OPTANH_H

#include "GPGOMEA/Operators/Operator.h"

class OpTanh : public Operator {
public:

    OpTanh() {
        arity = 1;
        name = "tanh";
        type = OperatorType::opFunction;
        is_arithmetic = false;
    }

    Operator * Clone() const override {
        return new OpTanh(*this);
    }

    arma::vec ComputeOutput(const arma::mat& x) override {
        return arma::tanh(x.col(0));
    }

    arma::vec Invert(const arma::vec& desired_elem, const arma::vec& output_siblings, size_t idx) override {

        // Valid elements are only within -1 and 1
        std::vector<double_t> res_v;
        res_v.reserve(desired_elem.n_elem);
        for (double_t v : desired_elem) {
            if (abs(v) <= 1.0){
                res_v.push_back(std::atanh(v));
                res_v.push_back(std::atanh(v)+2 * arma::datum::pi);
                res_v.push_back(std::atanh(v)-2 * arma::datum::pi);
            }
        }
        
        arma::vec res;
        if (res_v.empty()) {
            res = arma::vec(1);
            res[0] = arma::datum::inf;
        } else {
            res = arma::vec(res_v.size());
            for(size_t i = 0; i < res_v.size(); i++)
                res[i] = res_v[i];
        }
        return res;

    }

    std::string GetHumanReadableExpression(const std::vector<std::string>& args) override {
        return name + "(" + args[0] + ")";
    }

private:

};

#endif /* OPTANH_H */
